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
- 対角ゲートのみの区間は DiagBlock として分離し、置換なしの
  1 パス位相カーネルで適用する (散在ビットのまま適用できる)
- 区間分割は実測コスト比 (GEMM+置換 vs 位相パス) に基づく
  segment DP で決める
- complex64 の statevector も同じ経路で扱う。実表現 GEMM は
  sgemm になり、置換カーネルは Numba が dtype ごとに特殊化する

行列の添字規約は statevector.apply_gate / get_gate_matrix と同一:
ブロック qubit リストの先頭が行列インデックスの最上位ビット。
statevector は little-endian (qubit q = インデックスの bit q)。
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, NamedTuple, Tuple

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

# Segment DP のコスト (dense GEMM 1 ブロック = 1.0 に対する相対値)。
# 実測 (M3 Pro): dense = GEMM + 期待置換 0.5 パス、diag = 位相 1 パス。
# fp64: (9.05+2.55)/11.6 vs 4.63 → 0.40 / fp32: 6.2 vs 3.06 → 0.49。
_COST_DENSE = 1.0
_COST_DIAG = 0.45
_MAX_SEG_GATES = 4096   # DP の O(G^2) 抑止

# Accelerate は tall-skinny GEMM を単一クラスタでしか実行しないため、
# 行チャンクに分割して複数スレッドで dgemm/sgemm を並列発行する
# (BLAS は GIL を解放するので ThreadPool で足りる)。分割は行数基準の
# 3 段 (M3 Pro 実測、dgemm/sgemm 双方でほぼ最良):
#   rows <= 16K: 単発 (BLAS 固定費が支配的、分割は逆効果)
#   rows <  128K: 8 タスク
#   rows >= 128K: ~16K 行タスクへ過分割 (P/E コアの速度差を吸収;
#                 n=24 で dgemm +13% / sgemm +46%)
_SINGLE_GEMM_ROWS = 16384
_MID_GEMM_ROWS = 131072
_TARGET_CHUNK_ROWS = 16384
_POOL = None


class FusionStats:
    """Lightweight per-process counters for the fused apply path."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.b_dense = 0        # dense GEMM blocks applied
        self.b_diag = 0         # diagonal phase blocks applied
        self.n_perm = 0         # bit-permutation passes executed
        self.n_perm_skipped = 0  # permutations absorbed by |0..0> layout
        self.t_gemm = 0.0
        self.t_perm = 0.0
        self.t_phase = 0.0
        self.k_hist = {}

    def as_dict(self):
        return {
            'b_dense': self.b_dense, 'b_diag': self.b_diag,
            'n_perm': self.n_perm, 'n_perm_skipped': self.n_perm_skipped,
            't_gemm_ms': self.t_gemm * 1e3, 't_perm_ms': self.t_perm * 1e3,
            't_phase_ms': self.t_phase * 1e3, 'k_hist': dict(self.k_hist),
        }


stats = FusionStats()


class DenseBlock(NamedTuple):
    mat: np.ndarray          # (2^k, 2^k) complex128
    qubits: List[int]        # sorted descending; qubits[0] = MSB


class DiagBlock(NamedTuple):
    phases: np.ndarray       # (2^k,) complex128; bit j = j-th smallest qubit
    qubits: List[int]        # sorted descending (same convention as DenseBlock)


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(
            max_workers=min(8, os.cpu_count() or 4))
    return _POOL


def _matmul_rows_threaded(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    """out = a @ b, split over row chunks of ``a`` across threads.

    行数基準の adaptive 分割 (閾値はモジュール先頭のコメント参照):
    rows <= 16K は単発 GEMM、128K 未満は 8 タスク、以上は ~16K 行
    タスクへ過分割 (P/E 非対称のテール解消で n=24 dgemm +13%,
    sgemm +46% 実測)。
    """
    rows = a.shape[0]
    if rows <= _SINGLE_GEMM_ROWS:
        np.matmul(a, b, out=out)
        return
    if rows < _MID_GEMM_ROWS:
        ntask = 8
    else:
        ntask = min((rows + _TARGET_CHUNK_ROWS - 1) // _TARGET_CHUNK_ROWS,
                    256)
    chunk = (rows + ntask - 1) // ntask
    futures = [
        _pool().submit(np.matmul, a[s:s + chunk], b, out=out[s:s + chunk])
        for s in range(0, rows, chunk)
    ]
    for f in futures:
        f.result()


# ============================================================================
# Fusion: gate list -> list of DenseBlock / DiagBlock via segment DP
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


def _expand_diag(diag: np.ndarray,
                 gate_qubits: List[int],
                 block_qubits_desc: List[int]) -> np.ndarray:
    """Expand a gate's diagonal to the block's 2^k phase vector.

    The phase vector is indexed with the block's qubits in the same
    convention as DenseBlock matrices: block_qubits_desc[0] is the MSB,
    i.e. the j-th *smallest* block qubit is bit j of the index.
    """
    k = len(block_qubits_desc)
    kf = len(gate_qubits)
    m = np.arange(1 << k, dtype=np.int64)
    idx = np.zeros(1 << k, dtype=np.int64)
    for t, q in enumerate(gate_qubits):
        bitpos = k - 1 - block_qubits_desc.index(q)
        idx |= ((m >> bitpos) & 1) << (kf - 1 - t)
    return diag[idx]


# 常に対角なゲート名。行列値ではなく名前で判定することで、融合計画が
# パラメータ値に依存しなくなり、回路構造キーでキャッシュできる
# (crx(0)=I のような「特殊値でだけ対角」のケースは dense に落ちるが
# 正しさは変わらない)。
_DIAG_GATES = frozenset({
    'id', 'i', 'z', 's', 'sdg', 't', 'tdg', 'rz', 'p', 'u1',
    'cz', 'crz', 'cp', 'cu1', 'rzz', 'ccz',
})


class FusionPlan(NamedTuple):
    """Parameter-independent fusion plan for a circuit structure.

    bounds はフィルタ済みゲート列 (barrier 除去後) 上の区間
    [start, end) と種別。同じ構造 (ゲート名 + qubit 列) の回路なら
    パラメータ値が違っても再利用できる。
    """
    bounds: Tuple[Tuple[int, int, str], ...]   # (start, end, kind)


def _gate_fields(gate: Any) -> Tuple[str, List[int], List]:
    name = gate.name if hasattr(gate, 'name') else gate['name']
    qubits = gate.qubits if hasattr(gate, 'qubits') else gate['qubits']
    if hasattr(gate, 'params'):
        params = gate.params
    else:
        params = gate.get('params', [])
    return name, list(qubits), params


def _build_dense_block_numpy(items, union: List[int]) -> DenseBlock:
    """Numba-free fallback: incremental _expand_matrix + matmul chain."""
    cur_mat = None
    cur_q: List[int] = []
    for mat, qubits in items:
        if cur_mat is None:
            cur_q = sorted(set(qubits), reverse=True)
            cur_mat = _expand_matrix(mat, qubits, cur_q)
            continue
        u = sorted(set(cur_q) | set(qubits), reverse=True)
        block = _expand_matrix(cur_mat, cur_q, u)
        gate_full = _expand_matrix(mat, qubits, u)
        cur_mat = gate_full @ block
        cur_q = u
    if cur_q != union:
        cur_mat = _expand_matrix(cur_mat, cur_q, union)
    return DenseBlock(cur_mat, union)


def _build_dense_block(items) -> DenseBlock:
    union = sorted({q for _, qubits in items for q in qubits},
                   reverse=True)
    if not HAS_NUMBA:
        return _build_dense_block_numpy(items, union)
    # union を先に確定し、単位行列にゲートを順に作用させる (途中の
    # 再埋め込みが不要になり、1 ゲート = 1 numba カーネル呼び出し)。
    k = len(union)
    dim = 1 << k
    bitpos = {q: k - 1 - i for i, q in enumerate(union)}
    m = np.eye(dim, dtype=np.complex128)
    out = np.empty_like(m)
    for mat, qubits in items:
        kg = len(qubits)
        gpos = np.asarray([bitpos[qubits[kg - 1 - t]] for t in range(kg)],
                          dtype=np.int64)
        _apply_gate_to_block_numba(m, out,
                                   np.ascontiguousarray(mat), gpos, kg)
        m, out = out, m
    return DenseBlock(m, union)


def _build_diag_block(items) -> DiagBlock:
    union = sorted({q for _, qubits in items for q in qubits},
                   reverse=True)
    phases = np.ones(1 << len(union), dtype=np.complex128)
    for mat, qubits in items:
        phases *= _expand_diag(np.diagonal(mat).copy(), qubits, union)
    return DiagBlock(phases, union)


def _dense_reaches(items: List, max_fused: int) -> List[int]:
    """jd[i] = furthest exclusive end of a dense segment starting at i.

    スライディングウィンドウ (qubit 出現数カウンタ) で O(G)。
    max_fused より大きい単独ゲートは長さ 1 の区間として許す。
    """
    G = len(items)
    jd = [0] * G
    counts: dict = {}
    distinct = 0
    r = 0
    for i in range(G):
        if r < i:
            r = i
        while r < G and r - i < _MAX_SEG_GATES:
            uniq = set(items[r][1])
            new = sum(1 for q in uniq if counts.get(q, 0) == 0)
            if distinct + new > max_fused and r > i:
                break
            for q in uniq:
                counts[q] = counts.get(q, 0) + 1
            distinct += new
            r += 1
            if distinct > max_fused:
                break   # oversized single gate: segment is [i, i+1)
        jd[i] = r
        for q in set(items[i][1]):
            counts[q] -= 1
            if counts[q] == 0:
                distinct -= 1
    return jd


def _diag_reaches(items: List, max_fused: int) -> List[int]:
    """jg[i] = furthest exclusive end of an all-diagonal segment at i.

    jg[i] == i は「対角区間を開始できない」を意味する。
    """
    G = len(items)
    jg = [0] * G
    counts: dict = {}
    distinct = 0
    r = 0
    for i in range(G):
        if r < i:
            r = i
        if items[i][2]:
            while r < G and r - i < _MAX_SEG_GATES and items[r][2]:
                uniq = set(items[r][1])
                new = sum(1 for q in uniq if counts.get(q, 0) == 0)
                if distinct + new > max_fused:
                    break
                for q in uniq:
                    counts[q] = counts.get(q, 0) + 1
                distinct += new
                r += 1
            jg[i] = r
            if r > i:
                for q in set(items[i][1]):
                    counts[q] -= 1
                    if counts[q] == 0:
                        distinct -= 1
        else:
            jg[i] = i
    return jg


def make_plan(gates: List,
              num_qubits: int,
              max_fused: int = DEFAULT_MAX_FUSED_QUBITS
              ) -> FusionPlan:
    """Partition the gate list into DenseBlock / DiagBlock segments.

    最小コスト分割を segment DP で求める。dense 区間は union が
    max_fused を超えるまで、diag 区間は全ゲートが対角かつ union が
    max_fused 以下の間だけ伸ばせる。コストは実測比の定数
    (_COST_DENSE / _COST_DIAG)。max_fused より大きいゲートは単独の
    dense 区間になる。

    区間コストが長さに依らないので、DP は「開始 i から終端 e へ張れる
    ⟺ 到達列 j*(i) >= e」の区間辺グラフの最短路になる。到達列は
    スライディングウィンドウ、最小化は単調両端キューで、全体 O(G)。

    ゲート名と qubit 列だけに依存する (行列値は見ない) ので、返る
    FusionPlan は同一構造の回路で再利用できる (パラメータ非依存)。
    """
    max_fused = max(1, min(max_fused, num_qubits))
    items = []
    for gate in gates:
        name, qubits, params = _gate_fields(gate)
        if name == 'barrier':
            continue
        items.append((None, qubits, name in _DIAG_GATES))
    G = len(items)
    if G == 0:
        return FusionPlan(())

    jd = _dense_reaches(items, max_fused)
    jg = _diag_reaches(items, max_fused)

    from collections import deque
    dp = [0.0] + [float('inf')] * G
    back: List = [None] * (G + 1)   # back[e] = (start, kind)
    dqd: deque = deque()   # candidate starts, indices asc, dp asc
    dqg: deque = deque()

    for e in range(1, G + 1):
        i = e - 1
        while dqd and dp[dqd[-1]] >= dp[i]:
            dqd.pop()
        dqd.append(i)
        if jg[i] > i:
            while dqg and dp[dqg[-1]] >= dp[i]:
                dqg.pop()
            dqg.append(i)
        # jd/jg は単調非減少なので、失効候補はキュー先頭に集まる。
        while dqd and jd[dqd[0]] < e:
            dqd.popleft()
        while dqg and jg[dqg[0]] < e:
            dqg.popleft()
        best = dp[dqd[0]] + _COST_DENSE
        pick = (dqd[0], 'dense')
        if dqg:
            c = dp[dqg[0]] + _COST_DIAG
            if c < best:
                best = c
                pick = (dqg[0], 'diag')
        dp[e] = best
        back[e] = pick

    # Reconstruct segments.
    bounds = []
    e = G
    while e > 0:
        s, kind = back[e]
        bounds.append((s, e, kind))
        e = s
    bounds.reverse()
    return FusionPlan(tuple(bounds))


def build_blocks(gates: List, num_qubits: int, plan: FusionPlan) -> List:
    """Materialize a FusionPlan into DenseBlock / DiagBlock instances.

    行列の取得と合成だけを行う (計画は plan 済み)。パラメータが
    変わっても plan は共有し、こちらだけ評価ごとに呼ぶ。
    """
    if not plan.bounds:
        return []
    items = []
    for gate in gates:
        name, qubits, params = _gate_fields(gate)
        if name == 'barrier':
            continue
        items.append((get_gate_matrix(name, params), qubits))
    blocks = []
    for s, e, kind in plan.bounds:
        seg = items[s:e]
        if kind == 'diag':
            blocks.append(_build_diag_block(seg))
        else:
            blocks.append(_build_dense_block(seg))
    return blocks


def fuse_gates(gates: List,
               num_qubits: int,
               max_fused: int = DEFAULT_MAX_FUSED_QUBITS
               ) -> List:
    """Plan + build in one call (uncached convenience wrapper)."""
    return build_blocks(gates, num_qubits,
                        make_plan(gates, num_qubits, max_fused))


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

    @jit(nopython=True, parallel=True, cache=True)
    def _permute_bits_block_numba(src, dst, froms, tos, keep_mask,
                                  nmoved, r):
        """Bit permutation with the low r bits preserved.

        下位 r ビットが移動対象でない場合、src 側も 2^r 要素の連続
        ランになるためスライスコピーできる (キャッシュライン利用率が
        100% になり generic gather 比 r=2 で +15%, r>=3 で +30-60%)。
        前提: 全 froms[j] >= r かつ全 tos[j] >= r。
        """
        block = np.int64(1) << r
        nblocks = src.size >> r
        for bi in prange(nblocks):
            d = np.int64(bi) << r
            s = d & keep_mask
            for j in range(nmoved):
                s |= ((d >> tos[j]) & 1) << froms[j]
            dst[d:d + block] = src[s:s + block]

    @jit(nopython=True, parallel=True, cache=True)
    def _permute_bits_4_numba(src, dst, f0, t0, f1, t1, f2, t2, f3, t3,
                              keep_mask):
        """4-moved-bit specialization: 4 independent gathers per
        iteration to hide gather latency (memory-level parallelism;
        generic 比 +16-27%)。移動ビット数 4 は k=4 融合の最頻ケース。
        """
        nchunk = src.size >> 2
        for ci in prange(nchunk):
            d = np.int64(ci) << 2
            for k in range(4):
                dd = d + k
                s = (dd & keep_mask) \
                    | (((dd >> t0) & 1) << f0) \
                    | (((dd >> t1) & 1) << f1) \
                    | (((dd >> t2) & 1) << f2) \
                    | (((dd >> t3) & 1) << f3)
                dst[dd] = src[s]

    @jit(nopython=True, cache=True)
    def _apply_gate_to_block_numba(m, out, g, gpos, kg):
        """out = (g embedded at bit positions gpos) @ m.

        2^k x 2^k のブロック行列 m の行インデックスのうち gpos の
        ビット群にゲート g を作用させる (列はバッチ)。融合ブロック
        構築のホットパス: NumPy の tensordot/moveaxis ベースの展開
        (~12us/ゲート) を 1-2us に置き換える。

        g のインデックス規約は get_gate_matrix と同一
        (gate_qubits[0] が MSB)。gpos[t] は g のビット t (LSB=0) が
        載るブロック行ビット位置。
        """
        dim = m.shape[0]
        dg = 1 << kg
        mask = np.int64(0)
        for t in range(kg):
            mask |= np.int64(1) << gpos[t]
        rows = np.empty(dg, dtype=np.int64)
        for base in range(dim):
            if base & mask:
                continue
            for j in range(dg):
                r = np.int64(base)
                for t in range(kg):
                    if (j >> t) & 1:
                        r |= np.int64(1) << gpos[t]
                rows[j] = r
            for i in range(dg):
                ri = rows[i]
                for col in range(dim):
                    acc = 0.0 + 0.0j
                    for j in range(dg):
                        acc += g[i, j] * m[rows[j], col]
                    out[ri, col] = acc

    @jit(nopython=True, parallel=True, cache=True)
    def _apply_phases_numba(sv, out, phases, bits, nbits):
        """out[i] = sv[i] * phases[idx], idx assembled from ``bits``.

        bits[j] は j 番目に小さいブロック qubit の現在のビット位置。
        置換不要 (散在ビットのまま適用できる) で、out=sv の
        in-place 呼び出しも安全 (要素独立)。
        """
        for ii in prange(sv.size):
            i = np.int64(ii)
            idx = np.int64(0)
            for j in range(nbits):
                idx |= ((i >> bits[j]) & 1) << j
            out[i] = sv[i] * phases[idx]


def _real_rep(mat: np.ndarray, rdtype=np.float64) -> np.ndarray:
    """Real 2d x 2d representation of a complex matrix, acting on
    interleaved [re, im] pairs (i.e. a complex array viewed as its
    real dtype).

    Accelerate の dgemm/sgemm は zgemm/cgemm より速い (AMX の実行列
    経路) ため、複素 GEMM を同 FLOP 数の実 GEMM に変換して流す。
    """
    d = mat.shape[0]
    r = np.empty((2 * d, 2 * d), dtype=rdtype)
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


def _as_dense(block) -> Tuple[np.ndarray, List[int]]:
    if isinstance(block, DiagBlock):
        return np.diag(block.phases), block.qubits
    return block[0], block[1]


def _apply_blocks_transpose(sv: np.ndarray,
                            blocks: List,
                            num_qubits: int) -> np.ndarray:
    """Numba-free fallback: transpose block axes to the front + GEMM."""
    n = num_qubits
    for block in blocks:
        mat, q = _as_dense(block)
        mat = mat.astype(sv.dtype, copy=False)
        k = len(q)
        axes = [n - 1 - qi for qi in q]
        perm = axes + [ax for ax in range(n) if ax not in axes]
        t = sv.reshape([2] * n).transpose(perm)
        t = np.ascontiguousarray(t).reshape(1 << k, -1)
        out = (mat @ t).reshape([2] * n)
        sv = np.ascontiguousarray(out.transpose(np.argsort(perm)).reshape(-1))
    return sv


def _plan_permutation(pos_of: List[int], q_asc: List[int], k: int,
                      num_qubits: int):
    """Plan the one-pass permutation that moves ``q_asc`` into bits <k.

    ブロック qubit のうち既に下位ビットにいるものは現在位置を維持し、
    新規参入だけ空いた下位スロットへ入れる (移動ビット数の最小化;
    ビット順の食い違いは行列並べ替えで無料吸収)。追い出される qubit
    は空いた上位位置を昇順で埋める。
    """
    n = num_qubits
    block_set = set(q_asc)
    new_pos = list(pos_of)
    kept_slots = {pos_of[u] for u in q_asc if pos_of[u] < k}
    free_slots = [b for b in range(k) if b not in kept_slots]
    fi = 0
    for u in q_asc:
        if pos_of[u] >= k:
            new_pos[u] = free_slots[fi]
            fi += 1
    vacated = sorted(pos_of[u] for u in q_asc if pos_of[u] >= k)
    evictees = sorted(
        (u for u in range(n)
         if pos_of[u] < k and u not in block_set and pos_of[u] not in kept_slots),
        key=lambda u: pos_of[u])
    for u, p in zip(evictees, vacated):
        new_pos[u] = p
    return new_pos


def _run_permutation(sv, free, pos_of, new_pos, num_qubits):
    n = num_qubits
    froms = [pos_of[u] for u in range(n) if new_pos[u] != pos_of[u]]
    tos = [new_pos[u] for u in range(n) if new_pos[u] != pos_of[u]]
    keep_mask = (1 << n) - 1
    for t in tos:
        keep_mask ^= 1 << t
    fa = np.asarray(froms, dtype=np.int64)
    ta = np.asarray(tos, dtype=np.int64)
    # Dispatch (実測採用判定に基づく):
    #   下位 r>=2 ビット保存 -> 連続ブロックコピー置換
    #   移動 4 ビット        -> MLP 特殊化 (独立 gather x4)
    #   それ以外             -> generic gather
    r = min(min(froms), min(tos))
    if r >= 2:
        _permute_bits_block_numba(sv, free, fa, ta, keep_mask,
                                  len(froms), r)
    elif len(froms) == 4:
        _permute_bits_4_numba(sv, free,
                              fa[0], ta[0], fa[1], ta[1],
                              fa[2], ta[2], fa[3], ta[3], keep_mask)
    else:
        _permute_bits_numba(sv, free, fa, ta, keep_mask, len(froms))


def apply_fused_blocks(sv: np.ndarray,
                       blocks: List,
                       num_qubits: int,
                       assume_zero: bool = False,
                       return_layout: bool = False):
    """Apply fused blocks to the statevector.

    Never mutates the input array. Uses at most two scratch buffers of
    the statevector's size.

    Args:
        assume_zero: 入力が |0..0> (インデックス 0 のみ非ゼロ) である
            ことを保証する。ビット置換に対して不変なので、最初の dense
            GEMM までの置換はデータを動かさずレイアウト更新だけで済む。
        return_layout: 最終レイアウト復元を省略し (sv, pos_of) を返す。
            pos_of[q] = qubit q の状態が載っているインデックスビット。
    """
    n = num_qubits
    caller = sv
    sv = np.ascontiguousarray(sv)
    pos_of = list(range(n))
    if not blocks:
        # Keep the aliasing contract uniform: never hand back the
        # caller's own array.
        if sv is caller:
            sv = sv.copy()
        return (sv, pos_of) if return_layout else sv
    if not HAS_NUMBA:
        out = _apply_blocks_transpose(sv, blocks, n)
        return (out, pos_of) if return_layout else out

    free = np.empty_like(sv)
    # pristine: 状態がまだ c*|0..0> のまま (置換不変) かどうか
    pristine = assume_zero

    def next_free(prev):
        # The retired buffer is reusable unless it is the caller's array.
        return np.empty_like(caller) if prev is caller else prev

    for block in blocks:
        q = block.qubits if hasattr(block, 'qubits') else block[1]
        k = len(q)
        dim = 1 << k
        q_asc = sorted(q)

        if isinstance(block, DiagBlock):
            # 置換不要: 散在ビットのままインデックス位相パス 1 回。
            bits = np.asarray([pos_of[u] for u in q_asc], dtype=np.int64)
            ph = block.phases.astype(sv.dtype, copy=False)
            t0 = time.perf_counter()
            if sv is caller:
                _apply_phases_numba(sv, free, ph, bits, k)
                sv, free = free, next_free(sv)
            else:
                _apply_phases_numba(sv, sv, ph, bits, k)
            stats.t_phase += time.perf_counter() - t0
            stats.b_diag += 1
            stats.k_hist[k] = stats.k_hist.get(k, 0) + 1
            # 対角ブロックは |0..0> を c|0..0> に写すだけなので
            # pristine (置換不変性) は保たれる。
            continue

        mat = block.mat if hasattr(block, 'mat') else block[0]
        pos = [pos_of[u] for u in q_asc]
        hi = max(pos)

        if hi >= k:
            # Block not in the low bits: one parallel permutation pass
            # moves the block qubits there (already-low qubits stay put).
            new_pos = _plan_permutation(pos_of, q_asc, k, n)
            if pristine:
                # |0..0> はビット置換に不変: レイアウト更新のみ。
                stats.n_perm_skipped += 1
            else:
                t0 = time.perf_counter()
                _run_permutation(sv, free, pos_of, new_pos, n)
                stats.t_perm += time.perf_counter() - t0
                stats.n_perm += 1
                sv, free = free, next_free(sv)
            pos_of = new_pos
            pos = [pos_of[u] for u in q_asc]

        # Block occupies the low bits: tall-skinny GEMM. The complex
        # matvec runs as a real dgemm/sgemm on the real-dtype view (the
        # contracted axis is the interleaved one, so the real
        # representation applies directly).
        rdtype = np.float32 if sv.dtype == np.complex64 else np.float64
        m = _real_rep(_reorder_matrix(mat, pos), rdtype)
        t0 = time.perf_counter()
        _matmul_rows_threaded(
            sv.view(rdtype).reshape(-1, 2 * dim),
            np.ascontiguousarray(m.T),
            free.view(rdtype).reshape(-1, 2 * dim))
        stats.t_gemm += time.perf_counter() - t0
        stats.b_dense += 1
        stats.k_hist[k] = stats.k_hist.get(k, 0) + 1
        sv, free = free, next_free(sv)
        pristine = False

    if return_layout:
        if sv is caller:
            sv = sv.copy()
        return sv, pos_of

    if pos_of != list(range(n)):
        if pristine:
            stats.n_perm_skipped += 1
            if sv is caller:
                sv = sv.copy()
        else:
            t0 = time.perf_counter()
            _run_permutation(sv, free, pos_of,
                             list(range(n)), n)
            stats.t_perm += time.perf_counter() - t0
            stats.n_perm += 1
            sv = free
    elif sv is caller:
        sv = sv.copy()
    return sv


def apply_gates_fused(sv: np.ndarray,
                      gates: List,
                      num_qubits: int,
                      max_fused: int = DEFAULT_MAX_FUSED_QUBITS
                      ) -> np.ndarray:
    """Fuse the gate list and apply it to ``sv`` via Accelerate GEMMs."""
    blocks = fuse_gates(gates, num_qubits, max_fused)
    return apply_fused_blocks(sv, blocks, num_qubits)


def canonicalize_layout(sv: np.ndarray, pos_of: List[int],
                        num_qubits: int) -> np.ndarray:
    """Gather a layout statevector back to canonical qubit order.

    1 パスの NumPy fancy-index gather (Numba 不要)。canonical index の
    bit q は、置換レイアウトの bit pos_of[q] から来る。
    """
    if list(pos_of) == list(range(num_qubits)):
        return sv
    d = np.arange(sv.size, dtype=np.int64)
    src = np.zeros_like(d)
    for q, p in enumerate(pos_of):
        src |= ((d >> q) & 1) << p
    return sv[src]
