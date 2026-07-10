"""
tests/test_fusion_extras.py - CPU backend fusion extras

Covers the CPU backend features added on top of the base fusion path:
complex64 statevector/expectation, DiagBlock generation & application,
expectation_layout (matrix-free Pauli reduction on a possibly permuted
layout), the apply_fused_blocks(assume_zero=, return_layout=) API,
sampling remap through a lazy-permuted layout, the permutation-kernel
dispatch paths, and FusionStats counters.
"""
import numpy as np
import pytest

from metalq import Circuit
from metalq.backends.cpu.statevector import (
    initialize_statevector, apply_gates, expectation_from_statevector,
)
from metalq.backends.cpu.fusion import (
    apply_gates_fused, fuse_gates, apply_fused_blocks,
    DenseBlock, DiagBlock, stats,
)
from metalq.backends.cpu.expectation import expectation_layout
from metalq.backends.cpu.backend import CPUBackend
from metalq.spin import PauliTerm, Hamiltonian

from _circuit_helpers import random_circuit

rng = np.random.default_rng(20260710)


class _GateCircuit:
    """Minimal circuit-like object accepted by CPUBackend.

    Exposes exactly what CPUBackend touches (num_qubits, _gates,
    bind_parameters) so dict-form gate lists -- matching test_fusion.py's
    convention -- can be driven through the full backend API without
    building a real metalq.Circuit.
    """

    def __init__(self, num_qubits, gates):
        self.num_qubits = num_qubits
        self._gates = gates

    def bind_parameters(self, params):
        return self


def _random_circuit(n, depth):
    return random_circuit(rng, n, depth)


def _random_mixed_circuit(n, depth):
    """Random circuit weighted heavily towards diagonal gates (rz/rzz),
    mixed with dense single/two-qubit gates."""
    gates = []
    for _ in range(depth):
        r = rng.random()
        if r < 0.4:
            gates.append({'name': 'rz', 'qubits': [int(rng.integers(n))],
                          'params': [float(rng.uniform(-np.pi, np.pi))]})
        elif r < 0.55 and n >= 2:
            q = rng.choice(n, size=2, replace=False)
            gates.append({'name': 'rzz', 'qubits': [int(q[0]), int(q[1])],
                          'params': [float(rng.uniform(-np.pi, np.pi))]})
        elif r < 0.8:
            gates.append({'name': str(rng.choice(['h', 'x', 'y', 'sx'])),
                          'qubits': [int(rng.integers(n))], 'params': []})
        else:
            q = rng.choice(n, size=2, replace=False)
            gates.append({'name': str(rng.choice(['cx', 'cy', 'swap'])),
                          'qubits': [int(q[0]), int(q[1])], 'params': []})
    return gates


def _random_diag_circuit(n, depth):
    """Random circuit built exclusively from diagonal gates
    (rz/cz/cp/rzz/ccz/p/t/s)."""
    names_1q = ['rz', 'p', 't', 's']
    names_2q = ['cz', 'cp', 'rzz']
    gates = []
    for _ in range(depth):
        r = rng.random()
        if r < 0.5:
            name = str(rng.choice(names_1q))
            params = ([float(rng.uniform(-np.pi, np.pi))]
                      if name in ('rz', 'p') else [])
            gates.append({'name': name, 'qubits': [int(rng.integers(n))],
                          'params': params})
        elif r < 0.85 or n < 3:
            name = str(rng.choice(names_2q))
            q = rng.choice(n, size=2, replace=False)
            params = ([float(rng.uniform(-np.pi, np.pi))]
                      if name in ('cp', 'rzz') else [])
            gates.append({'name': name, 'qubits': [int(q[0]), int(q[1])],
                          'params': params})
        else:
            q = rng.choice(n, size=3, replace=False)
            gates.append({'name': 'ccz',
                          'qubits': [int(q[0]), int(q[1]), int(q[2])],
                          'params': []})
    return gates


def _random_hamiltonian(n, num_terms):
    terms = []
    for _ in range(num_terms):
        k = int(rng.integers(1, min(n, 3) + 1))
        qubits = rng.choice(n, size=k, replace=False)
        paulis = [str(rng.choice(['X', 'Y', 'Z'])) for _ in range(k)]
        ops = [(p, int(q)) for p, q in zip(paulis, qubits)]
        terms.append(PauliTerm(float(rng.uniform(-1, 1)), ops))
    return Hamiltonian(terms)


def _random_state(n, dtype=np.complex128):
    v = rng.standard_normal(1 << n) + 1j * rng.standard_normal(1 << n)
    v = v.astype(dtype)
    v /= np.linalg.norm(v)
    return v


def _to_canonical(sv, pos_of, n):
    """Manually reorder a lazy-permuted layout statevector back to
    canonical qubit order, mirroring apply_fused_blocks' internal
    final-restore permutation math."""
    idx = np.arange(1 << n, dtype=np.int64)
    perm_idx = np.zeros_like(idx)
    for q in range(n):
        bit_q = (idx >> q) & 1
        perm_idx |= bit_q << pos_of[q]
    return sv[perm_idx]


# ============================================================================
# 1. complex64 path
# ============================================================================

@pytest.mark.parametrize("n,depth", [(4, 30), (6, 40), (8, 25)])
def test_complex64_statevector_matches_c128(n, depth):
    gates = _random_circuit(n, depth)
    circ = _GateCircuit(n, gates)
    sv64 = CPUBackend(dtype=np.complex64).statevector(circ)
    sv128 = CPUBackend(dtype=np.complex128).statevector(circ)
    assert sv64.dtype == np.complex64
    assert sv128.dtype == np.complex128
    np.testing.assert_allclose(sv64.astype(np.complex128), sv128, atol=1e-5)


def test_complex64_expectation_matches_c128():
    n = 6
    gates = _random_circuit(n, 30)
    circ = _GateCircuit(n, gates)
    H = _random_hamiltonian(n, 5)
    e64 = CPUBackend(dtype=np.complex64).expectation(circ, H)
    e128 = CPUBackend(dtype=np.complex128).expectation(circ, H)
    assert abs(e64 - e128) < 1e-4


def test_complex64_requires_fusion():
    with pytest.raises(ValueError):
        CPUBackend(dtype=np.complex64, fusion=False)


def test_invalid_dtype_raises():
    with pytest.raises(ValueError):
        CPUBackend(dtype=np.float64)
    with pytest.raises(ValueError):
        CPUBackend(dtype=np.int32)


# ============================================================================
# 2. DiagBlock
# ============================================================================

def test_diag_only_circuit_produces_diagblocks():
    n = 6
    gates = _random_diag_circuit(n, 40)
    blocks = fuse_gates(gates, n, max_fused=4)
    assert len(blocks) > 0
    assert all(isinstance(b, DiagBlock) for b in blocks)


def test_diag_block_matches_einsum_reference():
    n = 6
    gates = _random_diag_circuit(n, 40)
    blocks = fuse_gates(gates, n, max_fused=4)
    ref = apply_gates(initialize_statevector(n), gates, n)
    got = apply_fused_blocks(initialize_statevector(n), blocks, n)
    np.testing.assert_allclose(got, ref, atol=1e-10)


def test_mixed_diag_dense_circuit_matches_reference():
    for _ in range(6):
        n = int(rng.integers(3, 9))
        depth = int(rng.integers(10, 60))
        gates = _random_mixed_circuit(n, depth)
        ref = apply_gates(initialize_statevector(n), gates, n)
        got = apply_gates_fused(initialize_statevector(n), gates, n,
                                max_fused=4)
        np.testing.assert_allclose(got, ref, atol=1e-10)


# ============================================================================
# 3. expectation_layout
# ============================================================================

def test_expectation_layout_matches_reference():
    n = 7
    gates = _random_circuit(n, 40)
    circ = _GateCircuit(n, gates)
    H = _random_hamiltonian(n, 6)

    got = CPUBackend(fusion=True).expectation(circ, H)

    sv_ref = CPUBackend(fusion=False).statevector(circ)
    ref = expectation_from_statevector(sv_ref, H, n)

    assert abs(got - ref) < 1e-9


def test_expectation_layout_single_pauliterm():
    n = 4
    gates = _random_circuit(n, 20)
    circ = _GateCircuit(n, gates)
    term = PauliTerm(0.7, [('X', 0), ('Y', 2), ('Z', 3)])

    got = CPUBackend(fusion=True).expectation(circ, term)

    sv_ref = CPUBackend(fusion=False).statevector(circ)
    ref = expectation_from_statevector(sv_ref, Hamiltonian([term]), n)

    assert abs(got - ref) < 1e-9


def test_expectation_layout_direct_call_matches_reference():
    # Exercise metalq.backends.cpu.expectation.expectation_layout directly
    # against a known-permuted layout (fusion's own _statevector_layout).
    n = 6
    gates = _random_circuit(n, 30)
    from metalq.backends.cpu.fusion import fuse_gates as _fuse
    blocks = _fuse(gates, n, max_fused=4)
    sv, pos_of = apply_fused_blocks(initialize_statevector(n), blocks, n,
                                    assume_zero=True, return_layout=True)
    H = _random_hamiltonian(n, 4)

    got = expectation_layout(sv, H, n, pos_of)

    sv_canon = apply_gates(initialize_statevector(n), gates, n)
    ref = expectation_from_statevector(sv_canon, H, n)
    assert abs(got - ref) < 1e-9


def test_expectation_repeated_pauli_ops_compose():
    # spin.py's PauliTerm @ / * concatenate ops without simplification, so
    # a term can carry several Paulis on the same qubit. The mask kernel
    # must compose them (Z.Z = I, Y.X = -iZ, Z.X.Z = -X, ...) exactly like
    # the sequential reference engine does.
    from metalq.spin import X, Y, Z
    n = 3
    gates = _random_circuit(n, 25)
    circ = _GateCircuit(n, gates)
    sv_ref = CPUBackend(fusion=False).statevector(circ)

    cases = [
        Z(0) @ Z(0),                    # I
        X(0) @ Y(0),                    # Y.X = -iZ(0)
        Z(1) @ X(1) @ Z(1),             # Z.X.Z = -X(1)
        Y(2) @ Z(2) @ X(0) @ X(0),      # (Z.Y on q2) x I on q0
        0.5 * (Z(0) @ Z(1)) + X(2) @ X(2) @ Y(0),
    ]
    for H in cases:
        got = CPUBackend(fusion=True).expectation(circ, H)
        ref_h = Hamiltonian([H]) if isinstance(H, PauliTerm) else H
        ref = expectation_from_statevector(sv_ref, ref_h, n)
        assert abs(got - ref) < 1e-9, f"mismatch for {H}"


# ============================================================================
# 4. apply_fused_blocks layout API
# ============================================================================

def test_return_layout_manual_reorder_matches_reference():
    n = 8
    gates = _random_circuit(n, 40)
    blocks = fuse_gates(gates, n, max_fused=4)
    sv0 = initialize_statevector(n)
    sv_layout, pos_of = apply_fused_blocks(sv0, blocks, n, assume_zero=True,
                                           return_layout=True)
    got = _to_canonical(sv_layout, pos_of, n)
    ref = apply_gates(initialize_statevector(n), gates, n)
    np.testing.assert_allclose(got, ref, atol=1e-10)


def test_assume_zero_matches_no_assume_zero():
    n = 8
    gates = _random_circuit(n, 40)
    blocks = fuse_gates(gates, n, max_fused=4)
    a = apply_fused_blocks(initialize_statevector(n), blocks, n,
                           assume_zero=True)
    b = apply_fused_blocks(initialize_statevector(n), blocks, n,
                           assume_zero=False)
    np.testing.assert_allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("assume_zero", [True, False])
@pytest.mark.parametrize("return_layout", [True, False])
def test_apply_fused_blocks_input_not_mutated_layout_flags(assume_zero,
                                                            return_layout):
    n = 7
    gates = _random_circuit(n, 30)
    blocks = fuse_gates(gates, n, max_fused=4)
    if assume_zero:
        sv = initialize_statevector(n)
    else:
        sv = _random_state(n)
    original = sv.copy()
    apply_fused_blocks(sv, blocks, n, assume_zero=assume_zero,
                       return_layout=return_layout)
    np.testing.assert_array_equal(sv, original)


# ============================================================================
# 5. sampling remap through a lazy-permuted layout
# ============================================================================

def test_sampling_remap_deterministic_high_qubits():
    # x(5); cx(7,6): qubit7 stays |0>, so cx is a no-op; the only
    # populated basis state is qubit5=1, everything else 0. Both gates
    # touch only high qubits (>= default max_fused_qubits=4), so the
    # fused blocks require a real bit permutation before sampling.
    n = 8
    c = Circuit(n)
    c.x(5)
    c.cx(7, 6)
    backend = CPUBackend(fusion=True)
    result = backend.run(c, shots=2000)
    counts = result['counts']
    expected = format(1 << 5, f'0{n}b')
    assert set(counts.keys()) == {expected}
    assert counts[expected] == 2000


def test_sampling_remap_support_matches_statevector():
    # h(5); cx(5,4) on high qubits produces a 2-outcome Bell-like state;
    # the sampled bitstring support must match the true (canonical)
    # statevector's support exactly.
    n = 6
    c = Circuit(n)
    c.h(5)
    c.cx(5, 4)
    backend = CPUBackend(fusion=True)
    result = backend.run(c, shots=4000)
    counts = result['counts']

    sv_ref = CPUBackend(fusion=False).statevector(c)
    support = {format(i, f'0{n}b') for i in range(len(sv_ref))
              if abs(sv_ref[i]) > 1e-9}

    assert set(counts.keys()) == support
    assert support == {'000000', '110000'}


# ============================================================================
# 6. permutation kernel dispatch paths
# ============================================================================

def _spy_kernel(monkeypatch, name):
    """Wrap a fusion permutation kernel to record that it was dispatched."""
    import metalq.backends.cpu.fusion as fusion_mod
    calls = []
    orig = getattr(fusion_mod, name)

    def wrapper(*args):
        calls.append(name)
        return orig(*args)

    monkeypatch.setattr(fusion_mod, name, wrapper)
    return calls


def test_permutation_dispatch_4bit_move(monkeypatch):
    # First block on {0,1,2,3} keeps the layout canonical; the second
    # block on {2,3,4,5} keeps qubits 2,3 in place, moves 4,5 into bits
    # 0,1 and evicts 0,1 to bits 4,5: froms=[4,5,0,1], tos=[0,1,4,5] --
    # exactly 4 moved bits with r=0, so the len(froms)==4 specialized
    # kernel must be dispatched.
    n = 8
    gates = [
        {'name': 'h', 'qubits': [0], 'params': []},
        {'name': 'cx', 'qubits': [0, 1], 'params': []},
        {'name': 'cy', 'qubits': [2, 3], 'params': []},
        {'name': 'h', 'qubits': [2], 'params': []},
        {'name': 'cx', 'qubits': [2, 4], 'params': []},
        {'name': 'swap', 'qubits': [3, 5], 'params': []},
        {'name': 'h', 'qubits': [4], 'params': []},
        {'name': 'cz', 'qubits': [4, 5], 'params': []},
    ]
    blocks = fuse_gates(gates, n, max_fused=4)
    assert sorted(blocks[-1].qubits) == [2, 3, 4, 5]

    calls = _spy_kernel(monkeypatch, '_permute_bits_4_numba')
    sv = _random_state(n)
    got = apply_fused_blocks(sv.copy(), blocks, n)
    assert calls, "expected the 4-moved-bit specialized kernel to fire"

    ref = apply_gates(sv.copy(), gates, n)
    np.testing.assert_allclose(got, ref, atol=1e-10)


def test_permutation_dispatch_low_bits_preserved(monkeypatch):
    # Block spans {0,1,5,6}: qubits 0,1 are already in the low bits and
    # stay put, only 5,6 move in -- the low 2 bits (positions 0,1) are
    # preserved, hitting the r>=2 block-copy permutation kernel.
    n = 8
    gates = [
        {'name': 'h', 'qubits': [0], 'params': []},
        {'name': 'cx', 'qubits': [0, 1], 'params': []},
        {'name': 'h', 'qubits': [5], 'params': []},
        {'name': 'cx', 'qubits': [5, 6], 'params': []},
        {'name': 'cz', 'qubits': [1, 6], 'params': []},
        {'name': 'swap', 'qubits': [0, 5], 'params': []},
        {'name': 'x', 'qubits': [6], 'params': []},
    ]
    blocks = fuse_gates(gates, n, max_fused=4)
    assert len(blocks) == 1
    assert sorted(blocks[0].qubits) == [0, 1, 5, 6]

    calls = _spy_kernel(monkeypatch, '_permute_bits_block_numba')
    sv = _random_state(n)
    got = apply_fused_blocks(sv.copy(), blocks, n)
    assert calls, "expected the block-copy permutation kernel to fire"

    ref = apply_gates(sv.copy(), gates, n)
    np.testing.assert_allclose(got, ref, atol=1e-10)


@pytest.mark.parametrize("trial", range(10))
def test_permutation_dispatch_random_sweep(trial):
    n = int(rng.integers(8, 11))
    depth = int(rng.integers(15, 50))
    gates = _random_circuit(n, depth)
    ref = apply_gates(initialize_statevector(n), gates, n)
    got = apply_gates_fused(initialize_statevector(n), gates, n,
                            max_fused=4)
    np.testing.assert_allclose(got, ref, atol=1e-10)


def test_plan_reuse_across_parameter_values():
    # A FusionPlan depends only on the circuit structure (names +
    # qubits): building blocks from the same plan with different
    # parameter values must match a from-scratch reference.
    from metalq.backends.cpu.fusion import make_plan, build_blocks
    n = 5

    def gates_for(thetas):
        gates = []
        for i, th in enumerate(thetas):
            gates.append({'name': 'ry', 'qubits': [i % n], 'params': [th]})
            gates.append({'name': 'rz', 'qubits': [(i + 1) % n],
                          'params': [-th]})
            gates.append({'name': 'cx', 'qubits': [i % n, (i + 2) % n],
                          'params': []})
        return gates

    plan = make_plan(gates_for([0.1] * 6), n, 4)
    for thetas in ([0.7, -1.2, 2.9, 0.0, 1.1, -0.4],
                   [np.pi] * 6):
        gates = gates_for(thetas)
        blocks = build_blocks(gates, n, plan)
        got = apply_fused_blocks(initialize_statevector(n), blocks, n)
        ref = apply_gates(initialize_statevector(n), gates, n)
        np.testing.assert_allclose(got, ref, atol=1e-10)


def test_backend_plan_cache_consistency():
    # Repeated expectation calls on same-structure circuits with
    # different parameters go through the backend plan cache; results
    # must keep matching the unfused reference.
    n = 4
    b = CPUBackend(fusion=True)
    H = _random_hamiltonian(n, 4)
    for _ in range(3):
        gates = []
        for q in range(n):
            gates.append({'name': 'ry', 'qubits': [q],
                          'params': [float(rng.uniform(-np.pi, np.pi))]})
        for q in range(n - 1):
            gates.append({'name': 'cx', 'qubits': [q, q + 1], 'params': []})
        circ = _GateCircuit(n, gates)
        got = b.expectation(circ, H)
        sv_ref = CPUBackend(fusion=False).statevector(circ)
        ref = expectation_from_statevector(sv_ref, H, n)
        assert abs(got - ref) < 1e-9


def test_empty_blocks_returns_fresh_array():
    # The aliasing contract is uniform: apply_fused_blocks never returns
    # the caller's own array, including the empty-blocks edge case.
    sv = initialize_statevector(4)
    out = apply_fused_blocks(sv, [], 4)
    assert out is not sv
    np.testing.assert_array_equal(out, sv)


def test_fuse_gates_linear_time_on_deep_small_circuit():
    # Planning must stay O(G): a deep circuit confined to few qubits used
    # to trigger O(G * _MAX_SEG_GATES) rescans (~28s for 20k gates).
    import time
    n = 4
    gates = [{'name': 'h', 'qubits': [i % n], 'params': []}
             for i in range(20000)]
    t0 = time.perf_counter()
    blocks = fuse_gates(gates, n, max_fused=4)
    dt = time.perf_counter() - t0
    assert len(blocks) <= 6      # merged into a handful of dense blocks
    assert dt < 3.0, f"fuse_gates took {dt:.1f}s on a 20k-gate circuit"


# ============================================================================
# 7. FusionStats
# ============================================================================

def test_fusion_stats_sanity():
    stats.reset()
    assert stats.b_dense == 0
    assert stats.b_diag == 0

    n = 6
    c = Circuit(n)
    for q in range(n):
        c.h(q)
    for q in range(n - 1):
        c.cx(q, q + 1)
    c.rz(0.3, 2)

    from metalq.spin import Z, X
    H = Z(0) @ Z(1) + 0.5 * X(2)

    CPUBackend(fusion=True).expectation(c, H)

    d = stats.as_dict()
    assert stats.b_dense + stats.b_diag >= 1
    assert stats.n_perm >= 0
    assert stats.n_perm_skipped >= 0
    assert d['t_gemm_ms'] >= 0.0
    assert d['t_perm_ms'] >= 0.0
    assert d['t_phase_ms'] >= 0.0
