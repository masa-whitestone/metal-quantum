"""
metalq/backends/cpu/adjoint.py - Adjoint Differentiation

可逆な adjoint 法による勾配計算 (PennyLane-Lightning 方式)。
parameter-shift の 2p 回の全回路実行に対し、

    forward 1 回 (フュージョン経路) + λ = H|ψ> 1 回
    + 逆順スイープ (ψ, λ を各ゲート 1 回ずつ巻き戻し)
    + パラメータゲートごとに 1 回の縮約パス

で全パラメータの勾配が得られる (回路適用 ~3 回相当)。

数式: E(θ) = <ψ|H|ψ>, ψ = U_L ... U_1 |0>。ゲート j のパラメータ v に
ついて dE/dv = 2 Re <λ_j | dU_j/dv | ψ_{j-1}> ここで
ψ_{j-1} = U_{j-1}...U_1|0>, λ_j = U_{j+1}†...U_L† H ψ。逆順に
ψ ← U_j†ψ, λ ← U_j†λ と巻き戻せば両者が同時に得られる。
dU/dv は μ = dU ψ を実体化せず <λ|dU|ψ> を 1 パスで直接縮約する。

回路レベルのパラメータ θ_i とゲートパラメータ v_g は
ParameterExpression.grad() の連鎖律で結び、dE/dθ_i = Σ_g (dv_g/dθ_i)
dE/dv_g とする (線形式のみ; 非線形や未対応ゲートは
AdjointUnsupported を投げて parameter-shift にフォールバック)。

λ は精度のため statevector の dtype に依らず complex128 で保持する
(部分和も float64)。
"""
from typing import List, TYPE_CHECKING

import numpy as np

from ...parameter import Parameter, ParameterExpression, is_parameterized
from .statevector import get_gate_matrix
from .expectation import _term_masks

if TYPE_CHECKING:
    from ...circuit import Circuit
    from ...spin import Hamiltonian
    from .backend import CPUBackend

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


class AdjointUnsupported(Exception):
    """Raised when the circuit cannot be differentiated by the adjoint
    method (caller should fall back to parameter-shift)."""


# ============================================================================
# Gate derivative matrices
# ============================================================================

_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_P1Q = {'x': _X, 'y': _Y, 'z': _Z}

# 単一パラメータの exp 形ゲートのみ対応 (u2/u3/r は fallback)
_DERIV_SUPPORTED = frozenset({
    'rx', 'ry', 'rz', 'p', 'u1',
    'crx', 'cry', 'crz', 'cp', 'cu1',
    'rxx', 'ryy', 'rzz', 'rzx',
})


def _deriv_matrix(name: str, params: List) -> np.ndarray:
    """dU/dv for a bound single-parameter gate.

    U(v) = exp(-i v G / 2) 形は dU = -i/2 G U。位相ゲート
    (p/cp: U = exp(i v |1..1><1..1|)) は dU = i P U。
    """
    U = get_gate_matrix(name, params)
    if name in ('rx', 'ry', 'rz'):
        return -0.5j * (_P1Q[name[1]] @ U)
    if name in ('crx', 'cry', 'crz'):
        G = np.zeros((4, 4), dtype=np.complex128)
        G[2:, 2:] = _P1Q[name[2]]
        return -0.5j * (G @ U)
    if name in ('rxx', 'ryy', 'rzz', 'rzx'):
        G = np.kron(_P1Q[name[1]], _P1Q[name[2]])
        return -0.5j * (G @ U)
    if name in ('p', 'u1'):
        return 1j * (np.diag([0.0, 1.0]).astype(np.complex128) @ U)
    if name in ('cp', 'cu1'):
        return 1j * (np.diag([0.0, 0.0, 0.0, 1.0]).astype(np.complex128) @ U)
    raise AdjointUnsupported(name)


# ============================================================================
# Numba kernels: in-place k-qubit gate application (k = 1, 2, 3),
# bra-op-ket contraction, and Pauli-term accumulation
# ============================================================================

if HAS_NUMBA:

    @jit(nopython=True, parallel=True, cache=True)
    def _apply_1q(sv, g, t):
        """sv <- (g on qubit t) sv, in place. g index: standard 2x2."""
        step = np.int64(1) << t
        half = sv.size >> 1
        for ii in prange(half):
            i = np.int64(ii)
            b = ((i >> t) << (t + 1)) | (i & (step - 1))
            i1 = b + step
            a0 = sv[b]
            a1 = sv[i1]
            sv[b] = g[0, 0] * a0 + g[0, 1] * a1
            sv[i1] = g[1, 0] * a0 + g[1, 1] * a1

    @jit(nopython=True, parallel=True, cache=True)
    def _apply_2q(sv, g, p1, p0):
        """sv <- g sv, in place. g row index = 2*bit(p1) + bit(p0)
        (p1 = qubits[0] = MSB of the gate index)."""
        lo = min(p0, p1)
        hi = max(p0, p1)
        s0 = np.int64(1) << p0
        s1 = np.int64(1) << p1
        mlo = (np.int64(1) << lo) - 1
        mhi = (np.int64(1) << hi) - 1
        quarter = sv.size >> 2
        for ii in prange(quarter):
            i = np.int64(ii)
            b = ((i >> lo) << (lo + 1)) | (i & mlo)
            b = ((b >> hi) << (hi + 1)) | (b & mhi)
            i1 = b + s0
            i2 = b + s1
            i3 = b + s0 + s1
            a0 = sv[b]
            a1 = sv[i1]
            a2 = sv[i2]
            a3 = sv[i3]
            sv[b] = g[0, 0] * a0 + g[0, 1] * a1 + g[0, 2] * a2 + g[0, 3] * a3
            sv[i1] = g[1, 0] * a0 + g[1, 1] * a1 + g[1, 2] * a2 + g[1, 3] * a3
            sv[i2] = g[2, 0] * a0 + g[2, 1] * a1 + g[2, 2] * a2 + g[2, 3] * a3
            sv[i3] = g[3, 0] * a0 + g[3, 1] * a1 + g[3, 2] * a2 + g[3, 3] * a3

    @jit(nopython=True, parallel=True, cache=True)
    def _apply_3q(sv, g, p2, p1, p0):
        """sv <- g sv, in place. g row index = 4*bit(p2)+2*bit(p1)+bit(p0)
        (p2 = qubits[0] = MSB)."""
        lo = min(p0, min(p1, p2))
        hi = max(p0, max(p1, p2))
        mid = (p0 + p1 + p2) - lo - hi
        s0 = np.int64(1) << p0
        s1 = np.int64(1) << p1
        s2 = np.int64(1) << p2
        mlo = (np.int64(1) << lo) - 1
        mmid = (np.int64(1) << mid) - 1
        mhi = (np.int64(1) << hi) - 1
        eighth = sv.size >> 3
        for ii in prange(eighth):
            i = np.int64(ii)
            b = ((i >> lo) << (lo + 1)) | (i & mlo)
            b = ((b >> mid) << (mid + 1)) | (b & mmid)
            b = ((b >> hi) << (hi + 1)) | (b & mhi)
            a0 = sv[b]
            a1 = sv[b + s0]
            a2 = sv[b + s1]
            a3 = sv[b + s1 + s0]
            a4 = sv[b + s2]
            a5 = sv[b + s2 + s0]
            a6 = sv[b + s2 + s1]
            a7 = sv[b + s2 + s1 + s0]
            for r in range(8):
                acc = (g[r, 0] * a0 + g[r, 1] * a1 + g[r, 2] * a2
                       + g[r, 3] * a3 + g[r, 4] * a4 + g[r, 5] * a5
                       + g[r, 6] * a6 + g[r, 7] * a7)
                idx = b
                if r & 1:
                    idx += s0
                if r & 2:
                    idx += s1
                if r & 4:
                    idx += s2
                sv[idx] = acc

    @jit(nopython=True, parallel=True, cache=True)
    def _brk_1q(lam, psi, g, t):
        """(re, im) of <lam| (g on qubit t) |psi>, float64 accumulation."""
        step = np.int64(1) << t
        half = psi.size >> 1
        re = 0.0
        im = 0.0
        for ii in prange(half):
            i = np.int64(ii)
            b = ((i >> t) << (t + 1)) | (i & (step - 1))
            i1 = b + step
            a0 = psi[b]
            a1 = psi[i1]
            m0 = g[0, 0] * a0 + g[0, 1] * a1
            m1 = g[1, 0] * a0 + g[1, 1] * a1
            l0 = lam[b]
            l1 = lam[i1]
            re += (np.float64(l0.real) * np.float64(m0.real)
                   + np.float64(l0.imag) * np.float64(m0.imag)
                   + np.float64(l1.real) * np.float64(m1.real)
                   + np.float64(l1.imag) * np.float64(m1.imag))
            im += (np.float64(l0.real) * np.float64(m0.imag)
                   - np.float64(l0.imag) * np.float64(m0.real)
                   + np.float64(l1.real) * np.float64(m1.imag)
                   - np.float64(l1.imag) * np.float64(m1.real))
        return re, im

    @jit(nopython=True, parallel=True, cache=True)
    def _brk_2q(lam, psi, g, p1, p0):
        """(re, im) of <lam| g |psi> for a 2-qubit operator."""
        lo = min(p0, p1)
        hi = max(p0, p1)
        s0 = np.int64(1) << p0
        s1 = np.int64(1) << p1
        mlo = (np.int64(1) << lo) - 1
        mhi = (np.int64(1) << hi) - 1
        quarter = psi.size >> 2
        re = 0.0
        im = 0.0
        for ii in prange(quarter):
            i = np.int64(ii)
            b = ((i >> lo) << (lo + 1)) | (i & mlo)
            b = ((b >> hi) << (hi + 1)) | (b & mhi)
            i1 = b + s0
            i2 = b + s1
            i3 = b + s0 + s1
            a0 = psi[b]
            a1 = psi[i1]
            a2 = psi[i2]
            a3 = psi[i3]
            for r in range(4):
                m = g[r, 0] * a0 + g[r, 1] * a1 + g[r, 2] * a2 + g[r, 3] * a3
                if r == 0:
                    l = lam[b]
                elif r == 1:
                    l = lam[i1]
                elif r == 2:
                    l = lam[i2]
                else:
                    l = lam[i3]
                re += (np.float64(l.real) * np.float64(m.real)
                       + np.float64(l.imag) * np.float64(m.imag))
                im += (np.float64(l.real) * np.float64(m.imag)
                       - np.float64(l.imag) * np.float64(m.real))
        return re, im

    @jit(nopython=True, parallel=True, cache=True)
    def _accumulate_pauli(psi, out, x_mask, sign_mask, w):
        """out += w * Q psi where (Q psi)[i] = sgn(i^x) psi[i^x],
        sgn(j) = (-1)^parity(j & sign_mask)."""
        for ii in prange(psi.size):
            i = np.int64(ii)
            j = i ^ x_mask
            v = np.int64(j & sign_mask)
            v ^= v >> 32
            v ^= v >> 16
            v ^= v >> 8
            v ^= v >> 4
            v ^= v >> 2
            v ^= v >> 1
            sgn = 1.0 - 2.0 * np.float64(v & 1)
            out[i] = out[i] + w * sgn * psi[j]


def _apply_gate_matrix(sv: np.ndarray, mat: np.ndarray, qubits: List[int]):
    """Apply a bound gate matrix to sv in place (1-3 qubits)."""
    g = np.ascontiguousarray(mat).astype(sv.dtype, copy=False)
    if len(qubits) == 1:
        _apply_1q(sv, g, qubits[0])
    elif len(qubits) == 2:
        _apply_2q(sv, g, qubits[0], qubits[1])
    elif len(qubits) == 3:
        _apply_3q(sv, g, qubits[0], qubits[1], qubits[2])
    else:
        raise AdjointUnsupported(f"{len(qubits)}-qubit gate")


def _bra_op_ket(lam: np.ndarray, psi: np.ndarray, mat: np.ndarray,
                qubits: List[int]) -> complex:
    g = np.ascontiguousarray(mat)
    if len(qubits) == 1:
        re, im = _brk_1q(lam, psi, g, qubits[0])
    elif len(qubits) == 2:
        re, im = _brk_2q(lam, psi, g, qubits[0], qubits[1])
    else:
        raise AdjointUnsupported(f"{len(qubits)}-qubit derivative gate")
    return complex(re, im)


def _apply_hamiltonian(psi: np.ndarray, hamiltonian,
                       num_qubits: int) -> np.ndarray:
    """λ = H_eff |psi> in complex128.

    E = Σ_t Re(c_t) Re<P_t> の勾配に効くのは各項のエルミート部分。
    P_t = w Q (w = i^{n_Y} × 合成位相, Q はマスク作用素) に対し
    Q† = (-1)^{n_Y} Q なのでエルミート部分は h Q,
    h = (w + conj(w) (-1)^{n_Y}) / 2。
    """
    lam = np.zeros(psi.size, dtype=np.complex128)
    for term in hamiltonian.terms:
        x_mask, sign_mask, n_y, phase = _term_masks(term)
        w = (1j ** (n_y & 3)) * phase
        h = (w + np.conj(w) * (-1.0) ** (n_y & 1)) / 2.0
        scalar = complex(term.coeff.real * h)
        if scalar == 0:
            continue
        _accumulate_pauli(psi, lam, np.int64(x_mask), np.int64(sign_mask),
                          scalar)
    return lam


# ============================================================================
# Adjoint gradient driver
# ============================================================================

def _analyze_parameters(circuit: 'Circuit'):
    """Map each non-barrier gate to its d(gate param)/d(theta_i) entries.

    Returns a list aligned with the non-barrier gate sequence; each
    element is a list of (theta_index, weight). Raises
    AdjointUnsupported for gates the adjoint method cannot handle.
    """
    plist = circuit.parameters
    pindex = {p: k for k, p in enumerate(plist)}
    dcoeffs = []
    for gate in circuit._gates:
        if gate.name == 'barrier':
            continue
        entries = []
        has_free = any(is_parameterized(p) for p in gate.params)
        if has_free:
            if gate.name not in _DERIV_SUPPORTED or len(gate.params) != 1:
                raise AdjointUnsupported(gate.name)
            pv = gate.params[0]
            if isinstance(pv, Parameter):
                entries.append((pindex[pv], 1.0))
            elif isinstance(pv, ParameterExpression):
                for p in pv.parameters:
                    try:
                        w = pv.grad(p)
                    except Exception as exc:
                        raise AdjointUnsupported(str(exc))
                    if w != 0.0:
                        entries.append((pindex[p], float(w)))
        dcoeffs.append(entries)
    return dcoeffs, len(plist)


def adjoint_gradient(backend: 'CPUBackend',
                     circuit: 'Circuit',
                     hamiltonian: 'Hamiltonian',
                     params: List[float]) -> np.ndarray:
    """Compute dE/dθ for all θ with ~3 circuit applications."""
    if not HAS_NUMBA:
        raise AdjointUnsupported("numba required")

    from ...spin import PauliTerm, Hamiltonian as _H
    if isinstance(hamiltonian, PauliTerm):
        hamiltonian = _H([hamiltonian])

    dcoeffs, num_params = _analyze_parameters(circuit)
    grads = np.zeros(num_params)
    if num_params == 0:
        return grads

    bound = circuit.bind_parameters(list(params))
    n = circuit.num_qubits
    gates = [g for g in bound._gates if g.name != 'barrier']

    # Forward via the fused path (canonical order), then λ = H ψ.
    # statevector() は呼び出し元と共有しないバッファを返す契約なので
    # in-place で巻き戻してよい。
    psi = backend.statevector(bound)
    lam = _apply_hamiltonian(psi, hamiltonian, n)

    # Reverse sweep.
    for j in range(len(gates) - 1, -1, -1):
        gate = gates[j]
        mat = get_gate_matrix(gate.name, gate.params)
        udag = np.conj(mat).T
        _apply_gate_matrix(psi, udag, gate.qubits)   # ψ -> ψ_{j-1}
        if dcoeffs[j]:
            dU = _deriv_matrix(gate.name, gate.params)
            val = 2.0 * _bra_op_ket(lam, psi, dU, gate.qubits).real
            for pi, w in dcoeffs[j]:
                grads[pi] += w * val
        if j > 0:
            _apply_gate_matrix(lam, udag, gate.qubits)

    return grads
