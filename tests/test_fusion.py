"""
tests/test_fusion.py - CPU backend gate fusion (fused GEMM path)

The fused Accelerate-GEMM path must produce statevectors identical to the
einsum reference implementation for all gate kinds, qubit orderings, and
fusion levels.
"""
import numpy as np
import pytest

from metalq.backends.cpu.statevector import (
    initialize_statevector, apply_gates,
)
from metalq.backends.cpu.fusion import (
    apply_gates_fused, fuse_gates, _expand_matrix,
)

rng = np.random.default_rng(1234)

SINGLE = ['x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'sx']
SINGLE_P = ['rx', 'ry', 'rz', 'p']
TWO = ['cx', 'cy', 'cz', 'ch', 'swap', 'iswap']
TWO_P = ['crx', 'cry', 'crz', 'cp', 'rxx', 'ryy', 'rzz', 'rzx']
THREE = ['ccx', 'ccz', 'cswap']


def _random_circuit(n, depth):
    gates = []
    for _ in range(depth):
        r = rng.random()
        if n < 3 and r >= 0.9:
            r = 0.5
        if r < 0.35:
            gates.append({'name': str(rng.choice(SINGLE)),
                          'qubits': [int(rng.integers(n))], 'params': []})
        elif r < 0.6:
            gates.append({'name': str(rng.choice(SINGLE_P)),
                          'qubits': [int(rng.integers(n))],
                          'params': [float(rng.uniform(-np.pi, np.pi))]})
        elif r < 0.75:
            q = rng.choice(n, size=2, replace=False)
            gates.append({'name': str(rng.choice(TWO)),
                          'qubits': [int(q[0]), int(q[1])], 'params': []})
        elif r < 0.9:
            q = rng.choice(n, size=2, replace=False)
            gates.append({'name': str(rng.choice(TWO_P)),
                          'qubits': [int(q[0]), int(q[1])],
                          'params': [float(rng.uniform(-np.pi, np.pi))]})
        else:
            q = rng.choice(n, size=3, replace=False)
            gates.append({'name': str(rng.choice(THREE)),
                          'qubits': [int(q[0]), int(q[1]), int(q[2])],
                          'params': []})
    return gates


@pytest.mark.parametrize("max_fused", [1, 2, 3, 4, 5])
def test_random_circuits_match_reference(max_fused):
    for _ in range(8):
        n = int(rng.integers(2, 10))
        depth = int(rng.integers(1, 100))
        gates = _random_circuit(n, depth)
        ref = apply_gates(initialize_statevector(n), gates, n)
        got = apply_gates_fused(initialize_statevector(n), gates, n,
                                max_fused=max_fused)
        np.testing.assert_allclose(got, ref, atol=1e-10)


def test_input_not_mutated():
    n = 6
    gates = _random_circuit(n, 40)
    sv = initialize_statevector(n)
    original = sv.copy()
    apply_gates_fused(sv, gates, n, max_fused=4)
    np.testing.assert_array_equal(sv, original)


def test_empty_and_barrier_only():
    n = 4
    sv0 = initialize_statevector(n)
    out = apply_gates_fused(sv0, [], n)
    np.testing.assert_array_equal(out, sv0)
    out = apply_gates_fused(
        sv0, [{'name': 'barrier', 'qubits': [], 'params': []}], n)
    np.testing.assert_array_equal(out, sv0)


def test_gate_larger_than_max_fused():
    # A 3-qubit gate must still work with max_fused=1.
    n = 4
    gates = [{'name': 'h', 'qubits': [0], 'params': []},
             {'name': 'ccx', 'qubits': [0, 1, 2], 'params': []}]
    ref = apply_gates(initialize_statevector(n), gates, n)
    got = apply_gates_fused(initialize_statevector(n), gates, n, max_fused=1)
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_fuse_reduces_block_count():
    gates = []
    for q in range(4):
        gates.append({'name': 'h', 'qubits': [q], 'params': []})
        gates.append({'name': 't', 'qubits': [q], 'params': []})
    blocks = fuse_gates(gates, 4, max_fused=4)
    assert len(blocks) == 1
    mat, q = blocks[0]
    assert mat.shape == (16, 16)
    assert q == [3, 2, 1, 0]


def test_expand_matrix_identity_embedding():
    # X on qubit 0 expanded into a {1,0} block must be I (x) X.
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    expanded = _expand_matrix(x, [0], [1, 0])
    np.testing.assert_allclose(expanded, np.kron(np.eye(2), x))
    # And into MSB position: X (x) I.
    expanded = _expand_matrix(x, [1], [1, 0])
    np.testing.assert_allclose(expanded, np.kron(x, np.eye(2)))


def test_cpu_backend_fusion_flag():
    from metalq.backends.cpu.backend import CPUBackend
    from metalq import Circuit

    c = Circuit(5)
    for q in range(5):
        c.h(q)
    for q in range(4):
        c.cx(q, q + 1)
    c.rz(1.234, 2)

    sv_fused = CPUBackend(fusion=True).statevector(c)
    sv_plain = CPUBackend(fusion=False).statevector(c)
    np.testing.assert_allclose(sv_fused, sv_plain, atol=1e-10)
