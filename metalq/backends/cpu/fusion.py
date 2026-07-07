"""
metalq/backends/cpu/fusion.py - Gate Fusion + GEMM Application

連続するゲートを最大 k qubit のブロックに融合し、ブロックごとに
1回の複素 GEMM で statevector に適用する。

macOS arm64 の NumPy は Accelerate BLAS にリンクされているため、
`sv.reshape(-1, 2^k) @ U.T` は zgemm として Apple の行列エンジン
(AMX / M4 以降は SME) 上で実行される。設計方針:

- 演算 (2^k x 2^k の行列適用) は必ず Accelerate GEMM に流す
- ブロックの qubit が下位ビットに揃っていない場合は、Numba の
  並列ビット置換カーネル (帯域律速の 1 パス) で下位に寄せる。
  置換状態 (qubit → ビット位置) は次のブロックへ持ち越し、
  最後に一度だけ元に戻す (lazy permutation)
- ビット順の食い違いは 2^k x 2^k 行列側の並べ替えで吸収する (無料)

行列の添字規約は statevector.apply_gate / get_gate_matrix と同一:
ブロック qubit リストの先頭が行列インデックスの最上位ビット。
statevector は little-endian (qubit q = インデックスの bit q)。
"""
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Tuple

import numpy as np

from .statevector import get_gate_matrix

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# 融合ブロックの最大 qubit 数。2^k x 2^k の GEMM になる。
# 大きいほどメモリパスは減るが、振幅あたりの FLOP は 2^k で増える。
DEFAULT_MAX_FUSED_QUBITS = 4

# Accelerate は tall-skinny GEMM を単一クラスタでしか実行しないため、
# 大きい statevector では行方向に分割して複数スレッドで zgemm/dgemm を
# 並列発行する (BLAS は GIL を解放するので ThreadPool で足りる)。
_THREAD_ELEMS = 1 << 19   # これ以上の要素数なら分割する
_POOL = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(
            max_workers=min(8, os.cpu_count() or 4))
    return _POOL


def _matmul_rows_threaded(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    """out = a @ b, split over row chunks of ``a`` across threads."""
    rows = a.shape[0]
    nt = min(8, os.cpu_count() or 4)
    if a.size < _THREAD_ELEMS or rows < nt:
        np.matmul(a, b, out=out)
        return
    chunk = (rows + nt - 1) // nt
    futures = [
        _pool().submit(np.matmul, a[s:s + chunk], b, out=out[s:s + chunk])
        for s in range(0, rows, chunk)
    ]
    for f in futures:
        f.result()


# ============================================================================
# Fusion: gate list -> list of (matrix, qubits) blocks
# ============================================================================

def _expand_matrix(mat: np.ndarray,
                   gate_qubits: List[int],
                   block_qubits: List[int]) -> np.ndarray:
    """Expand a gate matrix on ``gate_qubits`` to the block's full space.

    ``block_qubits`` must contain every qubit in ``gate_qubits``. The
    returned matrix is indexed with block_qubits[0] as the most
    significant bit (matching get_gate_matrix's convention for the
    gate's own qubit list).
    """
    kf = len(gate_qubits)
    kt = len(block_qubits)
    if kf == kt and list(gate_qubits) == list(block_qubits):
        return mat

    dim = 1 << kt
    # Treat the identity's columns as a batch of block statevectors and
    # apply the gate to the axes belonging to gate_qubits.
    eye = np.eye(dim, dtype=np.complex128).reshape([2] * kt + [dim])
    axes = [block_qubits.index(q) for q in gate_qubits]
    gate_tensor = mat.reshape([2] * (2 * kf))
    out = np.tensordot(gate_tensor, eye,
                       axes=(list(range(kf, 2 * kf)), axes))
    # tensordot leaves the gate's output axes in front; put each back at
    # its block position.
    out = np.moveaxis(out, range(kf), axes)
    return np.ascontiguousarray(out.reshape(dim, dim))


def _gate_fields(gate: Any) -> Tuple[str, List[int], List]:
    name = gate.name if hasattr(gate, 'name') else gate['name']
    qubits = gate.qubits if hasattr(gate, 'qubits') else gate['qubits']
    if hasattr(gate, 'params'):
        params = gate.params
    else:
        params = gate.get('params', [])
    return name, list(qubits), params


def fuse_gates(gates: List,
               num_qubits: int,
               max_fused: int = DEFAULT_MAX_FUSED_QUBITS
               ) -> List[Tuple[np.ndarray, List[int]]]:
    """Greedily fuse consecutive gates into blocks of <= max_fused qubits.

    Returns a list of (matrix, qubits) blocks where ``qubits`` is sorted
    descending and the matrix follows the same index convention as
    get_gate_matrix.
    """
    blocks: List[Tuple[np.ndarray, List[int]]] = []
    cur_mat = None
    cur_q: List[int] = []

    for gate in gates:
        name, qubits, params = _gate_fields(gate)
        if name == 'barrier':
            continue
        mat = get_gate_matrix(name, params)

        if cur_mat is None:
            cur_q = sorted(set(qubits), reverse=True)
            cur_mat = _expand_matrix(mat, qubits, cur_q)
            continue

        union = sorted(set(cur_q) | set(qubits), reverse=True)
        if len(union) <= max_fused:
            block = _expand_matrix(cur_mat, cur_q, union)
            gate_full = _expand_matrix(mat, qubits, union)
            cur_mat = gate_full @ block
            cur_q = union
        else:
            blocks.append((cur_mat, cur_q))
            cur_q = sorted(set(qubits), reverse=True)
            cur_mat = _expand_matrix(mat, qubits, cur_q)

    if cur_mat is not None:
        blocks.append((cur_mat, cur_q))
    return blocks


# ============================================================================
# Application: one GEMM per block, lazy bit permutation for scattered qubits
# ============================================================================

if HAS_NUMBA:

    @jit(nopython=True, parallel=True, cache=True)
    def _permute_bits_numba(src, dst, froms, tos, keep_mask, nmoved):
        """Bit-permute statevector indices: gather form.

        The qubit whose state lives at index bit froms[j] of ``src``
        lives at bit tos[j] of ``dst``. All other bits keep their
        position (keep_mask has those bits set).
        """
        size = src.size
        for di in prange(size):
            # np.int64: numba 0.63 parfors mistype a reassigned variable
            # seeded directly from the prange index (float64 inference).
            d = np.int64(di)
            s = d & keep_mask
            for j in range(nmoved):
                s |= ((d >> tos[j]) & 1) << froms[j]
            dst[d] = src[s]


def _real_rep(mat: np.ndarray) -> np.ndarray:
    """Real 2d x 2d representation of a complex matrix, acting on
    interleaved [re, im] pairs (i.e. a complex128 array viewed as
    float64).

    Accelerate の dgemm は zgemm より速い (AMX の fp64 実行列経路) ため、
    複素 GEMM を同 FLOP 数の実 GEMM に変換して流す。
    """
    d = mat.shape[0]
    r = np.empty((2 * d, 2 * d), dtype=np.float64)
    r[0::2, 0::2] = mat.real
    r[0::2, 1::2] = -mat.imag
    r[1::2, 0::2] = mat.imag
    r[1::2, 1::2] = mat.real
    return r


def _reorder_matrix(mat: np.ndarray, rel_pos: List[int]) -> np.ndarray:
    """Reindex a block matrix for a permuted bit layout.

    rel_pos[i] is the bit position (within the block's 2^k axis) that
    currently holds the i-th *smallest* block qubit; canonically that
    qubit belongs at bit i.
    """
    k = len(rel_pos)
    if list(rel_pos) == list(range(k)):
        return mat
    c = np.arange(1 << k, dtype=np.int64)
    m = np.zeros(1 << k, dtype=np.int64)
    for i, p in enumerate(rel_pos):
        m |= ((c >> p) & 1) << i
    return np.ascontiguousarray(mat[np.ix_(m, m)])


def _apply_blocks_transpose(sv: np.ndarray,
                            blocks: List[Tuple[np.ndarray, List[int]]],
                            num_qubits: int) -> np.ndarray:
    """Numba-free fallback: transpose block axes to the front + GEMM."""
    n = num_qubits
    for mat, q in blocks:
        k = len(q)
        axes = [n - 1 - qi for qi in q]
        perm = axes + [ax for ax in range(n) if ax not in axes]
        t = sv.reshape([2] * n).transpose(perm)
        t = np.ascontiguousarray(t).reshape(1 << k, -1)
        out = (mat @ t).reshape([2] * n)
        sv = np.ascontiguousarray(out.transpose(np.argsort(perm)).reshape(-1))
    return sv


def apply_fused_blocks(sv: np.ndarray,
                       blocks: List[Tuple[np.ndarray, List[int]]],
                       num_qubits: int) -> np.ndarray:
    """Apply fused blocks to the statevector, one GEMM per block.

    Never mutates the input array. Uses at most two scratch buffers of
    the statevector's size.
    """
    n = num_qubits
    sv = np.ascontiguousarray(sv)
    if not blocks:
        return sv
    if not HAS_NUMBA:
        return _apply_blocks_transpose(sv, blocks, n)

    caller = sv
    free = np.empty_like(sv)
    # pos_of[q] = index bit currently holding qubit q's state
    pos_of = list(range(n))

    def next_free(prev):
        # The retired buffer is reusable unless it is the caller's array.
        return np.empty_like(caller) if prev is caller else prev

    for mat, q in blocks:
        k = len(q)
        dim = 1 << k
        q_asc = sorted(q)
        pos = [pos_of[u] for u in q_asc]
        lo, hi = min(pos), max(pos)

        if hi >= k:
            # Block not in the low bits: one parallel permutation pass
            # moves the block qubits there. Qubits evicted from the low
            # bits take over the vacated positions.
            new_pos = list(pos_of)
            block_set = set(q_asc)
            for i, u in enumerate(q_asc):
                new_pos[u] = i
            vacated = sorted(p for p in pos if p >= k)
            evictees = sorted(
                (u for u in range(n)
                 if pos_of[u] < k and u not in block_set),
                key=lambda u: pos_of[u])
            for u, p in zip(evictees, vacated):
                new_pos[u] = p

            froms = [pos_of[u] for u in range(n) if new_pos[u] != pos_of[u]]
            tos = [new_pos[u] for u in range(n) if new_pos[u] != pos_of[u]]
            keep_mask = (1 << n) - 1
            for t in tos:
                keep_mask ^= 1 << t
            _permute_bits_numba(sv, free,
                                np.asarray(froms, dtype=np.int64),
                                np.asarray(tos, dtype=np.int64),
                                keep_mask, len(froms))
            sv, free = free, next_free(sv)
            pos_of = new_pos
            pos = [pos_of[u] for u in q_asc]

        # Block occupies the low bits: tall-skinny GEMM. The complex
        # matvec runs as a real dgemm on the float64 view (the contracted
        # axis is the interleaved one, so the real representation applies
        # directly). dgemm is ~1.6-2x faster than zgemm on Accelerate.
        m = _real_rep(_reorder_matrix(mat, pos))
        _matmul_rows_threaded(
            sv.view(np.float64).reshape(-1, 2 * dim),
            np.ascontiguousarray(m.T),
            free.view(np.float64).reshape(-1, 2 * dim))
        sv, free = free, next_free(sv)

    if pos_of != list(range(n)):
        froms = [pos_of[u] for u in range(n) if pos_of[u] != u]
        tos = [u for u in range(n) if pos_of[u] != u]
        keep_mask = (1 << n) - 1
        for t in tos:
            keep_mask ^= 1 << t
        _permute_bits_numba(sv, free,
                            np.asarray(froms, dtype=np.int64),
                            np.asarray(tos, dtype=np.int64),
                            keep_mask, len(froms))
        sv = free
    return sv


def apply_gates_fused(sv: np.ndarray,
                      gates: List,
                      num_qubits: int,
                      max_fused: int = DEFAULT_MAX_FUSED_QUBITS
                      ) -> np.ndarray:
    """Fuse the gate list and apply it to ``sv`` via Accelerate GEMMs."""
    max_fused = max(1, min(max_fused, num_qubits))
    blocks = fuse_gates(gates, num_qubits, max_fused)
    return apply_fused_blocks(sv, blocks, num_qubits)
