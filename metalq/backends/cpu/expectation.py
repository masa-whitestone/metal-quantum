"""
metalq/backends/cpu/expectation.py - Direct Pauli Expectation Reduction

<psi|P|psi> を **全項まとめて 1 パス**の Numba インデックスカーネルで
計算する。中間配列 (P|psi> の実体化) を作らず、complex64 state でも
部分和は float64 に積む (double reduction)。

パウリ作用の規約 (little-endian, qubit q = インデックスの bit q):
    P|j> = phase(j) * |j ^ x_mask>
    x_mask = X と Y のビット和
    phase(j) = i^{n_Y} * (-1)^{popcount(j & (y_mask | z_mask))}
よって
    <psi|P|psi> = sum_j conj(psi[j ^ x_mask]) * phase(j) * psi[j]

項は x_mask でグループ化してカーネルに渡す: gather psi[j ^ x] は
グループ内で共有され、x = 0 のグループ (Z のみの項; Ising 系では
全項) は |psi[j]|^2 の実数積だけで済む。統計は
(nchunks, nterms, 2) のチャンクローカル配列に積む (Numba の parfor
リダクションはスカラーしか畳めないため、prange はチャンク単位に
かけて内側を逐次ループにする)。

レイアウト置換された statevector (fusion の lazy permutation) には
pos_of で qubit -> ビット位置を写像してマスクを組めばよく、
canonical への復元パスが不要になる。

同じマスク表を使って lambda = H_eff |psi> も 1 書き込みパスで作れる
(adjoint 勾配の初期 costate; ``apply_hamiltonian_multi``)。
"""
from typing import NamedTuple

import numpy as np

try:
    from numba import jit, prange, get_num_threads
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def get_num_threads():      # pragma: no cover - numba absent
        return 4


# 1 スレッドあたりのチャンク数。多めに切って動的スケジューリングの
# 負荷分散を効かせつつ、チャンクローカル配列 (nchunks x nterms x 2)
# は数十 KB に収める。
_CHUNKS_PER_THREAD = 64


def _chunk_plan(size: int):
    """(nchunks, chunk) for the chunked-prange reduction kernel."""
    target = max(1, _CHUNKS_PER_THREAD * max(1, get_num_threads()))
    chunk = max(1, -(-size // target))
    return -(-size // chunk), chunk


if HAS_NUMBA:

    @jit(nopython=True, parallel=True, cache=True)
    def _pauli_sum_numba(sv, x_mask, sign_mask):
        """sum_j sign(j) * conj(sv[j ^ x_mask]) * sv[j], float64 に集約。

        sign(j) = (-1)^parity(j & sign_mask)。complex64 入力でも
        積和は float64 アキュムレータに積まれる。単一項用 (nterms == 1
        のときは multi 版よりわずかに速い)。
        """
        re = 0.0
        im = 0.0
        for jj in prange(sv.size):
            j = np.int64(jj)
            v = np.int64(j & sign_mask)
            v ^= v >> 32
            v ^= v >> 16
            v ^= v >> 8
            v ^= v >> 4
            v ^= v >> 2
            v ^= v >> 1
            sgn = 1.0 - 2.0 * np.float64(v & 1)
            a = sv[j ^ x_mask]
            b = sv[j]
            re += sgn * (np.float64(a.real) * np.float64(b.real)
                         + np.float64(a.imag) * np.float64(b.imag))
            im += sgn * (np.float64(a.real) * np.float64(b.imag)
                         - np.float64(a.imag) * np.float64(b.real))
        return re, im

    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def _pauli_sum_multi_numba(sv, x_masks, group_ptr, sign_masks,
                               nchunks, chunk, out):
        """All terms in ONE pass: out[c, t] = (re, im) partial sums.

        項は x_mask でグループ化済み (group_ptr[g]:group_ptr[g+1] が
        グループ g の項). 外側 prange をチャンクに、内側を逐次要素
        ループにして、チャンクローカルな (nterms, 2) アキュムレータに
        積む。チャンクは L1/L2 に収まるので、グループを外側に回して
        も statevector への DRAM トラフィックは 1 パスのままになる。
        """
        nterms = sign_masks.size
        ngroups = x_masks.size
        for c in prange(nchunks):
            lo = np.int64(c) * chunk
            hi = min(lo + chunk, np.int64(sv.size))
            acc = np.zeros((nterms, 2))
            for g in range(ngroups):
                x = x_masks[g]
                t0 = group_ptr[g]
                t1 = group_ptr[g + 1]
                if x == 0:
                    # 対角グループ: gather 不要、寄与は実数 |sv[j]|^2。
                    for j in range(lo, hi):
                        b = sv[j]
                        br = np.float64(b.real)
                        bi = np.float64(b.imag)
                        p = br * br + bi * bi
                        for t in range(t0, t1):
                            v = np.int64(j & sign_masks[t])
                            v ^= v >> 32
                            v ^= v >> 16
                            v ^= v >> 8
                            v ^= v >> 4
                            v ^= v >> 2
                            v ^= v >> 1
                            acc[t, 0] += (1.0 - 2.0 * np.float64(v & 1)) * p
                else:
                    for j in range(lo, hi):
                        b = sv[j]
                        a = sv[j ^ x]
                        ar = np.float64(a.real)
                        ai = np.float64(a.imag)
                        br = np.float64(b.real)
                        bi = np.float64(b.imag)
                        pr = ar * br + ai * bi
                        pi = ar * bi - ai * br
                        for t in range(t0, t1):
                            v = np.int64(j & sign_masks[t])
                            v ^= v >> 32
                            v ^= v >> 16
                            v ^= v >> 8
                            v ^= v >> 4
                            v ^= v >> 2
                            v ^= v >> 1
                            sgn = 1.0 - 2.0 * np.float64(v & 1)
                            acc[t, 0] += sgn * pr
                            acc[t, 1] += sgn * pi
            for t in range(nterms):
                out[c, t, 0] = acc[t, 0]
                out[c, t, 1] = acc[t, 1]

    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def _apply_pauli_multi_numba(psi, lam, x_masks, group_ptr, sign_masks,
                                 weights):
        """lam[i] = sum_t w_t sgn_t(i ^ x_t) psi[i ^ x_t]; returns Re<psi|lam>.

        1 書き込みパスで全項を合成する (項ごとの read-modify-write を
        やめる)。符号は ``_accumulate_pauli`` と同じく gather 先の
        インデックス j = i ^ x で評価する。エネルギー Re<psi|lam> は
        同じパスの中でスカラーリダクションとして取れる。
        """
        ngroups = x_masks.size
        energy = 0.0
        for ii in prange(psi.size):
            i = np.int64(ii)
            acc = 0.0 + 0.0j
            for g in range(ngroups):
                x = x_masks[g]
                j = i ^ x
                a = psi[j]
                w = 0.0 + 0.0j
                for t in range(group_ptr[g], group_ptr[g + 1]):
                    v = np.int64(j & sign_masks[t])
                    v ^= v >> 32
                    v ^= v >> 16
                    v ^= v >> 8
                    v ^= v >> 4
                    v ^= v >> 2
                    v ^= v >> 1
                    w += (1.0 - 2.0 * np.float64(v & 1)) * weights[t]
                acc += w * complex(np.float64(a.real), np.float64(a.imag))
            lam[i] = acc
            b = psi[i]
            energy += (np.float64(b.real) * acc.real
                       + np.float64(b.imag) * acc.imag)
        return energy


# sigma_new . sigma_cur -> (result, phase)。パウリ積の合成表
# (spin.py の PauliTerm は ops を簡約せず連結するため、同一 qubit に
# 複数の演算子が載った項をここで 1 つのパウリ + 位相に畳む)。
_PAULI_MUL = {
    ('X', 'X'): ('I', 1), ('Y', 'Y'): ('I', 1), ('Z', 'Z'): ('I', 1),
    ('X', 'Y'): ('Z', 1j), ('Y', 'X'): ('Z', -1j),
    ('Y', 'Z'): ('X', 1j), ('Z', 'Y'): ('X', -1j),
    ('Z', 'X'): ('Y', 1j), ('X', 'Z'): ('Y', -1j),
}


def _term_masks(term, pos_of=None):
    """Compose the term's Pauli string into masks + accumulated phase.

    ops は適用順 (先頭が最初に作用) なので、qubit ごとに左から
    乗算して合成する。合成後の演算子が P、位相が phase のとき
    元の項は phase * P。
    """
    composed = {}   # qubit -> current Pauli char ('I' は保持しない)
    phase = 1 + 0j
    for p_str, q in term.ops:
        if p_str not in ('X', 'Y', 'Z'):
            continue
        cur = composed.get(q, 'I')
        if cur == 'I':
            composed[q] = p_str
        else:
            res, ph = _PAULI_MUL[(p_str, cur)]
            phase *= ph
            if res == 'I':
                del composed[q]
            else:
                composed[q] = res

    x_mask = 0
    sign_mask = 0
    n_y = 0
    for q, p_str in composed.items():
        b = pos_of[q] if pos_of is not None else q
        if p_str == 'X':
            x_mask |= 1 << b
        elif p_str == 'Y':
            x_mask |= 1 << b
            sign_mask |= 1 << b
            n_y += 1
        else:   # 'Z'
            sign_mask |= 1 << b
    return x_mask, sign_mask, n_y, phase


class TermTables(NamedTuple):
    """Hamiltonian terms compiled into x_mask-grouped index tables.

    x_masks[g] はグループ g の X/Y ビットマスク、group_ptr[g]:
    group_ptr[g+1] がそのグループに属する項のスライス。項ごとに
    sign_mask (Z/Y ビット)、複素重み w = i^{n_Y} * 合成位相、実係数
    coeff、Y の個数 n_y を持つ (どれもグループ順に並ぶ)。
    """
    x_masks: np.ndarray       # (ngroups,) int64
    group_ptr: np.ndarray     # (ngroups + 1,) int64
    sign_masks: np.ndarray    # (nterms,) int64
    weights: np.ndarray       # (nterms,) complex128 -- i^n_Y * phase
    coeffs: np.ndarray        # (nterms,) float64 -- Re(term.coeff)
    n_y: np.ndarray           # (nterms,) int64


def build_term_tables(hamiltonian, pos_of=None) -> TermTables:
    """Compile a Hamiltonian into x_mask-grouped mask tables."""
    rows = []
    for term in hamiltonian.terms:
        x_mask, sign_mask, n_y, phase = _term_masks(term, pos_of)
        rows.append((x_mask, sign_mask, n_y,
                     (1j ** (n_y & 3)) * phase, float(term.coeff.real)))
    rows.sort(key=lambda r: r[0])

    x_masks = []
    group_ptr = [0]
    for i, r in enumerate(rows):
        if not x_masks or r[0] != x_masks[-1]:
            if x_masks:
                group_ptr.append(i)
            x_masks.append(r[0])
    group_ptr.append(len(rows))

    return TermTables(
        np.asarray(x_masks, dtype=np.int64),
        np.asarray(group_ptr, dtype=np.int64),
        np.asarray([r[1] for r in rows], dtype=np.int64),
        np.asarray([r[3] for r in rows], dtype=np.complex128),
        np.asarray([r[4] for r in rows], dtype=np.float64),
        np.asarray([r[2] for r in rows], dtype=np.int64),
    )


def hermitian_weights(tables: TermTables) -> np.ndarray:
    """Per-term scalar of the Hermitian part, Re(c_t) * h_t.

    E = sum_t Re(c_t) Re<P_t> の勾配 (と値) に効くのは各項の
    エルミート部分。P_t = w Q (Q はマスク作用素) に対し
    Q^dag = (-1)^{n_Y} Q なのでエルミート部分は h Q,
    h = (w + conj(w) (-1)^{n_Y}) / 2。
    """
    sgn = np.where(tables.n_y & 1, -1.0, 1.0)
    h = (tables.weights + np.conj(tables.weights) * sgn) / 2.0
    return tables.coeffs * h


def expectation_layout(sv: np.ndarray, hamiltonian, num_qubits: int,
                       pos_of=None) -> float:
    """Matrix-free <psi|H|psi> on a (possibly bit-permuted) statevector.

    全項を 1 パスで縮約する (項ごとの statevector 走査をやめる)。

    Args:
        pos_of: fusion の lazy permutation レイアウト
            (pos_of[q] = qubit q の状態が載っているインデックスビット)。
            None なら canonical。
    """
    if not HAS_NUMBA:
        # Numba なしフォールバック: canonical へ 1 パスで gather し、
        # 既存の逐次適用エンジンに委譲する (パウリ規約の実装を増やさ
        # ない)。complex64 は float64 蓄積の契約を守るため upcast。
        from .fusion import canonicalize_layout
        from .statevector import expectation_from_statevector
        if pos_of is not None:
            sv = canonicalize_layout(sv, pos_of, num_qubits)
        if sv.dtype != np.complex128:
            sv = sv.astype(np.complex128)
        return expectation_from_statevector(sv, hamiltonian, num_qubits)

    tables = build_term_tables(hamiltonian, pos_of)
    nterms = tables.sign_masks.size
    if nterms == 0:
        return 0.0

    if nterms == 1:
        re, im = _pauli_sum_numba(sv, np.int64(tables.x_masks[0]),
                                  np.int64(tables.sign_masks[0]))
        sums = np.asarray([complex(re, im)])
    else:
        nchunks, chunk = _chunk_plan(sv.size)
        out = np.zeros((nchunks, nterms, 2))
        _pauli_sum_multi_numba(sv, tables.x_masks, tables.group_ptr,
                               tables.sign_masks, nchunks, np.int64(chunk),
                               out)
        part = out.sum(axis=0)
        sums = part[:, 0] + 1j * part[:, 1]

    return float(np.sum(tables.coeffs * np.real(tables.weights * sums)))


def apply_hamiltonian_multi(psi: np.ndarray, hamiltonian, num_qubits: int,
                            pos_of=None):
    """lambda = H_eff |psi> (complex128) plus the energy Re<psi|lambda>.

    H_eff は各項のエルミート部分の和 (``hermitian_weights``); 反
    エルミート部分は <psi|H|psi> の虚部にしか効かないので、
    Re<psi|lambda> は expectation_layout と厳密に一致する。

    Returns:
        (energy, lam): energy は float、lam は complex128 の (2^n,)。
    """
    tables = build_term_tables(hamiltonian, pos_of)
    scalars = hermitian_weights(tables)

    # scalar == 0 の項は落とす (グループも空になれば消す)。
    keep = scalars != 0
    lam = np.zeros(psi.size, dtype=np.complex128)
    if not keep.any():
        return 0.0, lam

    x_masks = []
    group_ptr = [0]
    sign_masks = []
    weights = []
    for g in range(tables.x_masks.size):
        s, e = tables.group_ptr[g], tables.group_ptr[g + 1]
        idx = [t for t in range(s, e) if keep[t]]
        if not idx:
            continue
        x_masks.append(tables.x_masks[g])
        sign_masks.extend(int(tables.sign_masks[t]) for t in idx)
        weights.extend(complex(scalars[t]) for t in idx)
        group_ptr.append(len(sign_masks))

    x_masks = np.asarray(x_masks, dtype=np.int64)
    group_ptr = np.asarray(group_ptr, dtype=np.int64)
    sign_masks = np.asarray(sign_masks, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.complex128)

    if not HAS_NUMBA:
        return _apply_hamiltonian_numpy(psi, lam, x_masks, group_ptr,
                                        sign_masks, weights)

    energy = _apply_pauli_multi_numba(psi, lam, x_masks, group_ptr,
                                      sign_masks, weights)
    return float(energy), lam


def _apply_hamiltonian_numpy(psi, lam, x_masks, group_ptr, sign_masks,
                             weights):
    """Numba-free reference for ``apply_hamiltonian_multi``."""
    idx = np.arange(psi.size, dtype=np.int64)
    src = psi.astype(np.complex128, copy=False)
    for g in range(x_masks.size):
        j = idx ^ int(x_masks[g])
        w = np.zeros(psi.size, dtype=np.complex128)
        for t in range(group_ptr[g], group_ptr[g + 1]):
            v = j & int(sign_masks[t])
            for sh in (32, 16, 8, 4, 2, 1):
                v = v ^ (v >> sh)
            w += np.where(v & 1, -weights[t], weights[t])
        lam += w * src[j]
    energy = float(np.real(np.vdot(src, lam)))
    return energy, lam
