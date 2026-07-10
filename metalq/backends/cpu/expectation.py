"""
metalq/backends/cpu/expectation.py - Direct Pauli Expectation Reduction

<psi|P|psi> をパウリ文字列ごとに 1 パスの Numba インデックスカーネルで
計算する。中間配列 (P|psi> の実体化) を作らず、complex64 state でも
部分和は float64 に積む (double reduction)。

パウリ作用の規約 (little-endian, qubit q = インデックスの bit q):
    P|j> = phase(j) * |j ^ x_mask>
    x_mask = X と Y のビット和
    phase(j) = i^{n_Y} * (-1)^{popcount(j & (y_mask | z_mask))}
よって
    <psi|P|psi> = sum_j conj(psi[j ^ x_mask]) * phase(j) * psi[j]

レイアウト置換された statevector (fusion の lazy permutation) には
pos_of で qubit -> ビット位置を写像してマスクを組めばよく、
canonical への復元パスが不要になる。
"""
import numpy as np

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:

    @jit(nopython=True, parallel=True, cache=True)
    def _pauli_sum_numba(sv, x_mask, sign_mask):
        """sum_j sign(j) * conj(sv[j ^ x_mask]) * sv[j], float64 に集約。

        sign(j) = (-1)^parity(j & sign_mask)。complex64 入力でも
        積和は float64 アキュムレータに積まれる。
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


def expectation_layout(sv: np.ndarray, hamiltonian, num_qubits: int,
                       pos_of=None) -> float:
    """Matrix-free <psi|H|psi> on a (possibly bit-permuted) statevector.

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

    total = 0.0
    for term in hamiltonian.terms:
        x_mask, sign_mask, n_y, phase = _term_masks(term, pos_of)
        re, im = _pauli_sum_numba(sv, np.int64(x_mask), np.int64(sign_mask))
        s = complex(re, im) * (1j ** (n_y & 3)) * phase
        total += term.coeff.real * s.real
    return total
