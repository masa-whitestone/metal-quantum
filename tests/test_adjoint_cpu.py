"""
tests/test_adjoint_cpu.py - CPU adjoint differentiation & gradient contract

Covers the CPU backend's adjoint gradient (fused block reverse sweep),
the one-pass multi-term Pauli kernels it is built on, the fused
expectation_and_gradient path, the corrected parameter-shift fallback,
and the shared gate-slot -> circuit-parameter chain-rule helper.

References the adjoint gradient against
  * central finite differences of expectation(), and
  * ``adjoint_gradient_reference`` -- the original per-gate
    bra-op-ket sweep kept in adjoint.py for exactly this purpose.
"""
import numpy as np
import pytest

from metalq import Circuit, Parameter
from metalq.spin import PauliTerm, Hamiltonian
from metalq.backends.base import gate_slot_grads_to_parameter_grads
from metalq.backends.cpu.backend import CPUBackend
from metalq.backends.cpu import adjoint as adj
from metalq.backends.cpu.adjoint import (
    AdjointUnsupported, adjoint_gradient_reference, _apply_hamiltonian,
    _DERIV_SUPPORTED,
)
from metalq.backends.cpu.expectation import (
    expectation_layout, apply_hamiltonian_multi, build_term_tables,
    hermitian_weights, _apply_hamiltonian_numpy,
)
from metalq.backends.cpu.statevector import expectation_from_statevector

from _circuit_helpers import SINGLE, TWO, THREE

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

DTYPES = [np.complex128, np.complex64]

# Single-parameter gates the adjoint path differentiates natively.
P1Q = ['rx', 'ry', 'rz', 'p', 'u1']
P2Q = ['crx', 'cry', 'crz', 'cp', 'cu1', 'rxx', 'ryy', 'rzz', 'rzx']
assert set(P1Q) | set(P2Q) == set(_DERIV_SUPPORTED)


# ============================================================================
# Circuit / Hamiltonian builders
# ============================================================================

def random_param_circuit(rng, n, depth, reuse=True, exprs=True,
                         consts=True, three_qubit=True):
    """Random circuit mixing fixed gates and parameterized rotations.

    Slots are filled with bare Parameters, ``c * p`` / ``p + c``
    expressions, reused Parameters and plain floats (constant slots), so
    the whole gate-slot -> parameter chain rule gets exercised.
    """
    qc = Circuit(n)
    params = []

    def new_param():
        p = Parameter(f'p{len(params)}')
        params.append(p)
        return p

    def angle():
        r = rng.random()
        if consts and r < 0.2:
            return float(rng.uniform(-np.pi, np.pi))     # constant slot
        if reuse and params and r < 0.4:
            p = params[int(rng.integers(len(params)))]   # reused parameter
        else:
            p = new_param()
        if exprs and rng.random() < 0.4:
            c = float(rng.choice([2.0, -1.5, 0.5, 3.0]))
            return c * p if rng.random() < 0.7 else p + float(rng.uniform(-1, 1))
        return p

    for _ in range(depth):
        r = rng.random()
        if n >= 3 and three_qubit and r > 0.92:
            q = rng.choice(n, size=3, replace=False)
            qc._add_gate(str(rng.choice(THREE)), [int(x) for x in q])
        elif n >= 2 and r > 0.72:
            q = rng.choice(n, size=2, replace=False)
            if rng.random() < 0.55:
                qc._add_gate(str(rng.choice(P2Q)),
                             [int(q[0]), int(q[1])], [angle()])
            else:
                qc._add_gate(str(rng.choice(TWO)), [int(q[0]), int(q[1])])
        elif r > 0.35:
            qc._add_gate(str(rng.choice(P1Q)), [int(rng.integers(n))],
                         [angle()])
        else:
            qc._add_gate(str(rng.choice(SINGLE)), [int(rng.integers(n))])

    if not params:                      # guarantee at least one parameter
        p = new_param()
        qc.ry(p, 0)
    return qc, params


def random_hamiltonian(rng, n, nterms, identity=False, repeats=False):
    """Random Pauli Hamiltonian, optionally with an identity term and
    with several Paulis stacked on one qubit (X(0) @ Y(0) style)."""
    terms = []
    if identity:
        terms.append(PauliTerm(complex(rng.normal()), []))
    for _ in range(nterms):
        ops = []
        for q in range(n):
            if rng.random() < 0.45:
                ops.append((str(rng.choice(['X', 'Y', 'Z'])), q))
                if repeats and rng.random() < 0.3:
                    ops.append((str(rng.choice(['X', 'Y', 'Z'])), q))
        terms.append(PauliTerm(complex(rng.normal()), ops))
    return Hamiltonian(terms)


def ising(n):
    t = [PauliTerm(1.0, [('Z', q)]) for q in range(n)]
    t += [PauliTerm(0.5, [('Z', q), ('Z', q + 1)]) for q in range(n - 1)]
    return Hamiltonian(t)


def qaoa_circuit(n=3):
    """QAOA-style circuit: 2*gamma / 2*beta expressions driving several
    gates each, plus a constant-valued rz slot."""
    gamma, beta = Parameter('gamma'), Parameter('beta')
    qc = Circuit(n)
    for q in range(n):
        qc.h(q)
    for q in range(n - 1):
        qc.rz(2 * gamma, q)
    qc.rz(0.3, n - 1)                      # constant slot
    for q in range(n):
        qc.rx(2 * beta, q)
    return qc, [gamma, beta]


def central_diff(backend, circuit, ham, vals, eps=1e-6):
    g = np.zeros(len(vals))
    for i in range(len(vals)):
        hi = list(vals)
        hi[i] += eps
        lo = list(vals)
        lo[i] -= eps
        g[i] = (backend.expectation(circuit, ham, hi)
                - backend.expectation(circuit, ham, lo)) / (2 * eps)
    return g


# ============================================================================
# A. one-pass multi-term Pauli reduction
# ============================================================================

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('nterms', [1, 2, 9, 25])
def test_multi_term_expectation_matches_single_terms(dtype, nterms):
    """One-pass reduction == sum of per-term reductions == einsum ref."""
    rng = np.random.default_rng(4242 + nterms)
    n = 6
    sv = (rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n))
    sv /= np.linalg.norm(sv)
    sv = sv.astype(dtype)

    H = random_hamiltonian(rng, n, nterms, identity=True, repeats=True)

    got = expectation_layout(sv, H, n)
    per_term = sum(expectation_layout(sv, Hamiltonian([t]), n)
                   for t in H.terms)
    ref = expectation_from_statevector(sv.astype(np.complex128), H, n)

    tol = 1e-10 if dtype == np.complex128 else 2e-5
    assert got == pytest.approx(per_term, abs=tol)
    assert got == pytest.approx(ref, abs=tol)


@pytest.mark.parametrize('dtype', DTYPES)
def test_multi_term_expectation_permuted_layout(dtype):
    """Masks built through pos_of reproduce the canonical-layout value."""
    rng = np.random.default_rng(77)
    n = 6
    size = 1 << n
    sv = (rng.normal(size=size) + 1j * rng.normal(size=size))
    sv /= np.linalg.norm(sv)
    sv = sv.astype(dtype)
    H = random_hamiltonian(rng, n, 8, identity=True, repeats=True)
    ref = expectation_layout(sv, H, n)

    pos_of = list(rng.permutation(n))
    idx = np.arange(size, dtype=np.int64)
    src = np.zeros(size, dtype=np.int64)
    for q, p in enumerate(pos_of):
        src |= ((idx >> p) & 1) << q     # canonical index of layout index
    sv_layout = sv[src]

    got = expectation_layout(sv_layout, H, n, pos_of)
    tol = 1e-10 if dtype == np.complex128 else 2e-5
    assert got == pytest.approx(ref, abs=tol)


@pytest.mark.parametrize('dtype', DTYPES)
def test_apply_hamiltonian_multi_matches_accumulation(dtype):
    """One-pass λ = H_eff ψ == the per-term accumulation, and the energy
    it returns == expectation()."""
    rng = np.random.default_rng(909)
    n = 6
    size = 1 << n
    sv = (rng.normal(size=size) + 1j * rng.normal(size=size))
    sv /= np.linalg.norm(sv)
    sv = sv.astype(dtype)
    H = random_hamiltonian(rng, n, 10, identity=True, repeats=True)

    energy, lam = apply_hamiltonian_multi(sv, H, n)
    lam_ref = _apply_hamiltonian(sv, H, n)
    tol = 1e-10 if dtype == np.complex128 else 2e-5
    assert np.allclose(lam, lam_ref, atol=tol)
    assert energy == pytest.approx(expectation_layout(sv, H, n), abs=tol)

    # Numba-free fallback agrees too.
    tables = build_term_tables(H)
    lam0 = np.zeros(size, dtype=np.complex128)
    e2, lam2 = _apply_hamiltonian_numpy(
        sv, lam0, tables.x_masks, tables.group_ptr, tables.sign_masks,
        hermitian_weights(tables).astype(np.complex128))
    assert np.allclose(lam2, lam_ref, atol=tol)
    assert e2 == pytest.approx(energy, abs=tol)


def test_apply_hamiltonian_multi_permuted_layout():
    """λ built on a permuted layout equals the permuted canonical λ."""
    rng = np.random.default_rng(1717)
    n = 5
    size = 1 << n
    sv = (rng.normal(size=size) + 1j * rng.normal(size=size))
    sv /= np.linalg.norm(sv)
    H = random_hamiltonian(rng, n, 6, identity=True)
    _, lam_ref = apply_hamiltonian_multi(sv, H, n)

    pos_of = list(rng.permutation(n))
    idx = np.arange(size, dtype=np.int64)
    src = np.zeros(size, dtype=np.int64)
    for q, p in enumerate(pos_of):
        src |= ((idx >> p) & 1) << q
    energy, lam = apply_hamiltonian_multi(sv[src], H, n, pos_of)
    assert np.allclose(lam, lam_ref[src], atol=1e-10)
    assert energy == pytest.approx(expectation_layout(sv, H, n), abs=1e-10)


def test_zero_scalar_terms_are_dropped():
    """A term whose Hermitian part vanishes contributes nothing to λ."""
    n = 3
    rng = np.random.default_rng(5)
    sv = (rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n))
    sv /= np.linalg.norm(sv)
    # X(0) @ Y(0) composes to i*Z(0): anti-Hermitian part only once the
    # real coefficient is taken, so H_eff drops it.
    H = Hamiltonian([PauliTerm(1.0, [('X', 0), ('Y', 0)])])
    energy, lam = apply_hamiltonian_multi(sv, H, n)
    assert np.allclose(lam, 0.0)
    assert energy == pytest.approx(0.0, abs=1e-12)
    assert expectation_layout(sv, H, n) == pytest.approx(0.0, abs=1e-12)


# ============================================================================
# B. adjoint gradient: reference + finite differences
# ============================================================================

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('n', [2, 3, 5, 8])
def test_adjoint_matches_reference_sweep(dtype, n):
    """Fused block sweep == the per-gate bra-op-ket reference sweep."""
    rng = np.random.default_rng(1000 + n)
    qc, ps = random_param_circuit(rng, n, depth=6 * n)
    H = random_hamiltonian(rng, n, 5, identity=True, repeats=True)
    vals = list(rng.uniform(-np.pi, np.pi, len(ps)))
    backend = CPUBackend(dtype=dtype)

    got = backend.gradient(qc, H, vals)
    ref = adjoint_gradient_reference(backend, qc, H, vals)
    assert got.shape == (len(ps),)
    tol = 1e-9 if dtype == np.complex128 else 1e-4
    assert np.allclose(got, ref, atol=tol), np.abs(got - ref).max()


@pytest.mark.parametrize('n', [2, 3, 4, 5, 6])
def test_adjoint_matches_finite_differences(n):
    rng = np.random.default_rng(2000 + n)
    qc, ps = random_param_circuit(rng, n, depth=4 * n)
    H = random_hamiltonian(rng, n, 4, identity=True, repeats=True)
    vals = list(rng.uniform(-2, 2, len(ps)))
    backend = CPUBackend()

    got = backend.gradient(qc, H, vals)
    fd = central_diff(backend, qc, H, vals)
    assert np.allclose(got, fd, atol=1e-6), np.abs(got - fd).max()


@pytest.mark.parametrize('name', P1Q + P2Q)
def test_every_supported_gate_gradient(name):
    """Each _DERIV_SUPPORTED gate differentiates correctly, both as a bare
    parameter and inside a ``2 * p`` expression."""
    n = 3
    p, q = Parameter('a'), Parameter('b')
    qc = Circuit(n)
    qc.h(0)
    qc.h(1)
    qc.rx(0.4, 2)
    qubits = [0, 1] if name in P2Q else [1]
    qc._add_gate(name, qubits, [p])
    qc.cx(0, 2)
    qc._add_gate(name, qubits, [2 * q])
    qc.ry(0.7, 0)
    H = Hamiltonian([PauliTerm(1.0, [('Z', 0), ('X', 1)]),
                     PauliTerm(-0.6, [('Y', 2)])])
    vals = [0.63, -1.11]
    backend = CPUBackend()

    got = backend.gradient(qc, H, vals)
    assert np.allclose(got, central_diff(backend, qc, H, vals), atol=1e-6)
    assert np.allclose(got, adjoint_gradient_reference(backend, qc, H, vals),
                       atol=1e-9)


def test_parameter_reused_across_gates_and_expressions():
    """One parameter driving several gates (bare and scaled) accumulates."""
    a = Parameter('a')
    qc = Circuit(3)
    qc.h(0)
    qc.ry(a, 0)
    qc.rz(2 * a, 1)
    qc.cx(0, 1)
    qc.rx(-1.5 * a, 2)
    qc.rzz(a + 0.25, 1, 2)
    qc.rz(0.9, 0)                              # constant slot
    H = ising(3)
    backend = CPUBackend()
    vals = [0.37]
    got = backend.gradient(qc, H, vals)
    assert got.shape == (1,)
    assert np.allclose(got, central_diff(backend, qc, H, vals), atol=1e-6)
    assert np.allclose(got, adjoint_gradient_reference(backend, qc, H, vals),
                       atol=1e-9)


def test_block_with_several_parameterized_gates():
    """A block whose qubit union stays <= K holds many parameterized gates;
    each one's derivative must come back separately."""
    ps = [Parameter(f'p{i}') for i in range(6)]
    qc = Circuit(2)
    qc.h(0)
    qc.h(1)
    for i, p in enumerate(ps):
        (qc.rx if i % 3 == 0 else qc.ry if i % 3 == 1 else qc.rz)(p, i % 2)
    qc.rzz(0.5, 0, 1)
    H = Hamiltonian([PauliTerm(1.0, [('X', 0)]), PauliTerm(0.8, [('Z', 1)])])
    backend = CPUBackend()
    vals = list(np.linspace(-1.0, 1.0, len(ps)))

    blocks = adj._plan_reverse_blocks(
        [g for g in qc.bind_parameters(vals)._gates], adj.BLOCK_QUBITS)
    assert max(len(b.gate_idx) for b in blocks) > 1     # really fused

    got = backend.gradient(qc, H, vals)
    assert np.allclose(got, central_diff(backend, qc, H, vals), atol=1e-6)


def test_three_qubit_gates_inside_blocks():
    a, b = Parameter('a'), Parameter('b')
    qc = Circuit(4)
    for q in range(4):
        qc.h(q)
    qc.ry(a, 0)
    qc.ccx(0, 1, 2)
    qc.rz(2 * b, 3)
    qc.cswap(1, 2, 3)
    qc.rx(b, 1)
    qc.ccz(0, 2, 3)
    qc.crz(a, 2, 3)
    H = Hamiltonian([PauliTerm(1.0, [('Z', 1), ('Z', 2)]),
                     PauliTerm(0.4, [('X', 3)])])
    backend = CPUBackend()
    vals = [0.8, -0.45]
    got = backend.gradient(qc, H, vals)
    assert np.allclose(got, central_diff(backend, qc, H, vals), atol=1e-6)
    assert np.allclose(got, adjoint_gradient_reference(backend, qc, H, vals),
                       atol=1e-9)


@pytest.mark.parametrize('dtype', DTYPES)
def test_gradient_on_permuted_layout(dtype):
    """The reverse sweep runs on fusion's lazy-permuted layout; make sure a
    circuit that really permutes still differentiates correctly."""
    n = 6
    rng = np.random.default_rng(31337)
    qc, ps = random_param_circuit(rng, n, depth=30, three_qubit=False)
    vals = list(rng.uniform(-1, 1, len(ps)))
    backend = CPUBackend(dtype=dtype)
    _, pos_of = backend._statevector_layout(qc.bind_parameters(vals))
    assert pos_of != list(range(n)), "expected a non-canonical layout"

    H = ising(n)
    got = backend.gradient(qc, H, vals)
    ref = adjoint_gradient_reference(backend, qc, H, vals)
    tol = 1e-9 if dtype == np.complex128 else 1e-4
    assert np.allclose(got, ref, atol=tol)


def test_gradient_without_parameters():
    qc = Circuit(2)
    qc.h(0)
    qc.cx(0, 1)
    backend = CPUBackend()
    assert backend.gradient(qc, ising(2), []).shape == (0,)
    energy, grad = backend.expectation_and_gradient(qc, ising(2), [])
    assert grad.shape == (0,)
    assert energy == pytest.approx(backend.expectation(qc, ising(2), []),
                                   rel=1e-12)


def test_fusion_disabled_backend_gradient():
    """The adjoint path also works when the fused forward is turned off."""
    rng = np.random.default_rng(6)
    qc, ps = random_param_circuit(rng, 4, depth=14, three_qubit=False)
    vals = list(rng.uniform(-1, 1, len(ps)))
    H = ising(4)
    got = CPUBackend(fusion=False).gradient(qc, H, vals)
    assert np.allclose(got, CPUBackend().gradient(qc, H, vals), atol=1e-9)


# ============================================================================
# C. expectation_and_gradient
# ============================================================================

@pytest.mark.parametrize('dtype', DTYPES)
def test_expectation_and_gradient_agrees_with_separate_calls(dtype):
    rng = np.random.default_rng(808)
    n = 6
    qc, ps = random_param_circuit(rng, n, depth=25)
    H = random_hamiltonian(rng, n, 6, identity=True, repeats=True)
    vals = list(rng.uniform(-1.5, 1.5, len(ps)))
    backend = CPUBackend(dtype=dtype)

    energy, grad = backend.expectation_and_gradient(qc, H, vals)
    assert energy == pytest.approx(backend.expectation(qc, H, vals),
                                   rel=1e-10, abs=1e-12)
    assert np.allclose(grad, backend.gradient(qc, H, vals), atol=1e-12)


def test_expectation_and_gradient_falls_back():
    """A parameterized u3 forces the parameter-shift fallback; the energy
    must still be right and the gradient must match finite differences."""
    a, b = Parameter('a'), Parameter('b')
    qc = Circuit(2)
    qc.h(0)
    qc._add_gate('u3', [0], [a, 0.3, 2 * b])
    qc.cx(0, 1)
    qc.ry(b, 1)
    H = Hamiltonian([PauliTerm(1.0, [('Z', 0)]), PauliTerm(0.5, [('X', 1)])])
    backend = CPUBackend()
    vals = [0.4, -0.9]

    with pytest.raises(AdjointUnsupported):
        adj.adjoint_energy_and_gradient(backend, qc, H, vals)

    energy, grad = backend.expectation_and_gradient(qc, H, vals)
    assert energy == pytest.approx(backend.expectation(qc, H, vals), rel=1e-12)
    assert np.allclose(grad, central_diff(backend, qc, H, vals), atol=1e-6)
    assert np.allclose(grad, backend.gradient(qc, H, vals), atol=1e-12)


def test_nonlinear_expression_is_refused():
    """Non-linear expressions have no linear chain-rule coefficient: the
    adjoint path refuses them and the shift-rule fallback raises rather
    than returning a wrong gradient."""
    a, b = Parameter('a'), Parameter('b')
    qc = Circuit(1)
    qc.h(0)
    qc.rz(a * b, 0)
    backend = CPUBackend()
    H = Hamiltonian([PauliTerm(1.0, [('X', 0)])])
    with pytest.raises(AdjointUnsupported):
        adj.adjoint_energy_and_gradient(backend, qc, H, [0.5, 0.25])
    with pytest.raises(NotImplementedError, match="cannot differentiate"):
        backend.gradient(qc, H, [0.5, 0.25])
    # The energy itself is unaffected.
    assert np.isfinite(backend.expectation(qc, H, [0.5, 0.25]))


# ============================================================================
# D. gradient contract: parameter-shift + chain-rule helper
# ============================================================================

def test_parameter_shift_on_qaoa_expressions():
    """The shift rule applies to gate parameter SLOTS: shifting the circuit
    parameter of an rz(2*gamma) by pi/2 moves the gate angle by pi and
    silently returns 0 (the bug this rewrite fixes)."""
    qc, ps = qaoa_circuit(3)
    H = Hamiltonian([PauliTerm(1.0, [('Z', 0), ('Z', 1)]),
                     PauliTerm(0.7, [('X', 2)])])
    backend = CPUBackend()
    vals = [0.31, -0.77]

    shift = backend.gradient(qc, H, vals, method='parameter_shift')
    fd = central_diff(backend, qc, H, vals)
    assert shift.shape == (2,)
    assert np.allclose(shift, fd, atol=1e-6), (shift, fd)
    assert np.allclose(shift, backend.gradient(qc, H, vals), atol=1e-9)
    assert abs(shift[0]) > 1e-3          # the old code returned exactly 0


def test_parameter_shift_matches_finite_differences():
    rng = np.random.default_rng(4)
    n = 3
    # Only 2-eigenvalue generators here: controlled rotations need the
    # 4-term rule and are deliberately out of scope for the shift path.
    qc = Circuit(n)
    a, b, c = Parameter('a'), Parameter('b'), Parameter('c')
    qc.h(0)
    qc.rx(a, 0)
    qc.ry(2 * b, 1)
    qc.rz(0.75, 2)                      # constant slot
    qc.cx(0, 1)
    qc.rzz(c, 1, 2)
    qc.p(b, 2)
    H = random_hamiltonian(rng, n, 4, identity=True)
    backend = CPUBackend()
    vals = [0.2, -0.6, 1.1]
    got = backend.gradient(qc, H, vals, method='parameter_shift')
    assert np.allclose(got, central_diff(backend, qc, H, vals), atol=1e-6)


def test_parameter_shift_skips_constant_slots():
    """Constant slots cost no circuit evaluations."""
    calls = []
    backend = CPUBackend()
    real_expectation = backend.expectation

    def counting(circuit, ham, params=None):
        calls.append(1)
        return real_expectation(circuit, ham, params)

    backend.expectation = counting
    qc = Circuit(2)
    a = Parameter('a')
    qc.rx(a, 0)
    qc.rz(0.5, 1)          # constant slot: no shift evaluations
    qc.cx(0, 1)
    backend.gradient(qc, ising(2), [0.3], method='parameter_shift')
    assert len(calls) == 2


def test_chain_rule_helper_parameters_expressions_constants():
    a, b = Parameter('a'), Parameter('b')
    qc = Circuit(2)
    qc.rz(2 * a, 0)        # slot 0
    qc.rz(2 * a, 1)        # slot 1
    qc.rx(2 * b, 0)        # slot 2
    qc.rz(0.3, 1)          # slot 3 (constant)
    qc.ry(a, 0)            # slot 4
    qc.cx(0, 1)            # no slots
    assert qc.parameters == [a, b]

    out = gate_slot_grads_to_parameter_grads(qc, [1.0, 1.0, 1.0, 5.0, 1.0])
    assert np.allclose(out, [2 + 2 + 1, 2])

    out = gate_slot_grads_to_parameter_grads(qc, np.zeros(5))
    assert np.allclose(out, [0.0, 0.0])


def test_chain_rule_helper_offset_expression_and_no_params():
    a = Parameter('a')
    qc = Circuit(1)
    qc.rx(a + 0.5, 0)      # d/da = 1
    qc.rz(1.25, 0)         # constant
    assert np.allclose(gate_slot_grads_to_parameter_grads(qc, [3.0, 9.0]),
                       [3.0])

    plain = Circuit(1)
    plain.h(0)
    assert gate_slot_grads_to_parameter_grads(plain, []).shape == (0,)


def test_chain_rule_helper_length_mismatch():
    a = Parameter('a')
    qc = Circuit(1)
    qc.rx(a, 0)
    qc.rz(0.2, 0)
    with pytest.raises(ValueError, match="expected 2"):
        gate_slot_grads_to_parameter_grads(qc, [1.0])
    with pytest.raises(ValueError, match="expected 2"):
        gate_slot_grads_to_parameter_grads(qc, [1.0, 2.0, 3.0])


def test_gradient_length_is_unique_parameter_count():
    """Regression: more gate parameter slots than unique parameters."""
    qc, ps = qaoa_circuit(4)
    nslots = sum(len(g.params) for g in qc.gates)
    assert nslots > len(qc.parameters)
    backend = CPUBackend()
    for method in ('adjoint', 'parameter_shift'):
        assert backend.gradient(qc, ising(4), [0.2, 0.5],
                                method=method).shape == (2,)
    assert backend.expectation_and_gradient(
        qc, ising(4), [0.2, 0.5])[1].shape == (2,)


# ============================================================================
# torch integration
# ============================================================================

def test_quantum_layer_backward_matches_cpu_gradient():
    pytest.importorskip('torch')
    from metalq.torch.layer import QuantumLayer

    qc, ps = qaoa_circuit(3)
    H = Hamiltonian([PauliTerm(1.0, [('Z', 0), ('Z', 1)]),
                     PauliTerm(0.7, [('X', 2)])])
    layer = QuantumLayer(qc, H, backend_name='cpu')
    loss = layer()
    loss.backward()

    vals = layer.weights.detach().numpy().tolist()
    ref = CPUBackend().gradient(qc, H, vals)
    assert layer.weights.grad.shape == (len(qc.parameters),)
    assert np.allclose(layer.weights.grad.numpy(), ref, atol=1e-5)
    assert float(loss) == pytest.approx(
        CPUBackend().expectation(qc, H, vals), abs=1e-5)


@pytest.mark.parametrize('name', P1Q + P2Q)
def test_unwind_deriv_matrix_is_parameter_independent(name):
    """adjoint.py caches Dm = (dU/dv) U^dag by gate NAME, which is only
    valid because every _DERIV_SUPPORTED gate is a one-parameter
    exponential family U(v) = exp(v A): then dU/dv = A U and Dm = A.
    A new gate that breaks this must not be added to _DERIV_SUPPORTED
    without revisiting the cache."""
    from metalq.backends.cpu.adjoint import _deriv_matrix
    from metalq.backends.cpu.statevector import get_gate_matrix

    def dm(v):
        u = get_gate_matrix(name, [v])
        return _deriv_matrix(name, [v]) @ np.conj(u).T

    ref = dm(0.37)
    for v in (-2.4, 0.0, 1.9, 3.0):
        assert np.allclose(dm(v), ref, atol=1e-12), v
