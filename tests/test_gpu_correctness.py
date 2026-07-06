"""
Regression tests for GPU backend correctness and the GPU expectation path.

Covers bugs fixed in the GPU-optimization pass:
- 3-qubit gates (ccx/cswap/ccz) were silently skipped on the MPS backend and
  applied with inverted bit-order on the CPU einsum fallback.
- Phase gates (s/t/sdg/tdg/p) were silently mapped to identity on MPS.
- 2-qubit gates other than CX (cy/cz/swap/cp) were identity on MPS.
- spin.to_matrix used big-endian qubit order while both backends simulate
  little-endian, corrupting every expectation value.
- The GPU inner-product reduction kernel dropped contributions (barrier-less
  warp-synchronous tail), corrupting adjoint gradients and GPU expectations.
"""
import numpy as np
import pytest

from metalq import Circuit
from metalq.backends.cpu.backend import CPUBackend
from metalq.spin import X, Y, Z, Hamiltonian

try:
    from metalq.backends.mps.backend import MPSBackend
    _mps = MPSBackend()
    HAS_MPS = _mps.is_available()
except Exception:
    HAS_MPS = False

cpu = CPUBackend()

needs_mps = pytest.mark.skipif(not HAS_MPS, reason="MPS backend not available")


def _all_gates_circuit():
    c = Circuit(4)
    for q in range(4):
        c.h(q)
    c.s(0); c.t(1); c.sdg(2); c.tdg(3)
    c.p(0.3, 0); c.rx(0.5, 1); c.ry(0.7, 2); c.rz(0.9, 3)
    c.cy(0, 1); c.cz(1, 2); c.swap(2, 3); c.cp(0.4, 0, 3)
    c.cx(0, 2); c.ccx(0, 1, 2); c.cswap(1, 2, 3); c.ccz(0, 2, 3)
    return c


class TestEndianness:
    """Statevector convention is little-endian (qubit 0 = LSB, Qiskit style)."""

    def test_x0_statevector_index(self):
        c = Circuit(2); c.x(0)
        sv = cpu.statevector(c)
        assert np.argmax(np.abs(sv)) == 1  # |01> = index 1

    def test_to_matrix_matches_statevector_convention(self):
        # After x(0), <Z(0)> must be -1. Fails if to_matrix is big-endian.
        c = Circuit(2); c.x(0)
        assert cpu.expectation(c, 1.0 * Z(0)) == pytest.approx(-1.0)
        assert cpu.expectation(c, 1.0 * Z(1)) == pytest.approx(+1.0)

    def test_ccx_little_endian(self):
        # |q0=1,q1=1,q2=0> --CCX(0,1,2)--> |q0=1,q1=1,q2=1> (index 3 -> 7)
        c = Circuit(3); c.x(0); c.x(1); c.ccx(0, 1, 2)
        sv = cpu.statevector(c)
        assert np.argmax(np.abs(sv)) == 7


@needs_mps
class TestMPSGateParity:
    """Every gate type must produce the same state on MPS and CPU."""

    @pytest.mark.parametrize("name,apply", [
        ("s",     lambda c: c.s(1)),
        ("t",     lambda c: c.t(2)),
        ("sdg",   lambda c: c.sdg(1)),
        ("tdg",   lambda c: c.tdg(2)),
        ("p",     lambda c: c.p(0.4, 1)),
        ("cy",    lambda c: c.cy(0, 2)),
        ("cz",    lambda c: c.cz(1, 3)),
        ("swap",  lambda c: c.swap(1, 3)),
        ("cp",    lambda c: c.cp(0.4, 0, 3)),
        ("ccx",   lambda c: c.ccx(0, 1, 2)),
        ("cswap", lambda c: c.cswap(1, 2, 3)),
        ("ccz",   lambda c: c.ccz(0, 2, 3)),
    ])
    def test_gate_matches_cpu(self, name, apply):
        c = Circuit(4)
        for q in range(4):
            c.h(q)
        # break symmetry so endianness errors can't cancel
        c.rz(0.3, 0); c.ry(0.5, 1); c.rz(0.7, 2); c.ry(0.9, 3)
        apply(c)
        np.testing.assert_allclose(
            _mps.statevector(c), cpu.statevector(c), atol=1e-5,
            err_msg=f"Gate {name} differs between MPS and CPU")

    def test_phase_gate_not_identity(self):
        # s(0) on |+> must actually change the state (was identity on GPU)
        c0 = Circuit(1); c0.h(0)
        c1 = Circuit(1); c1.h(0); c1.s(0)
        assert np.max(np.abs(_mps.statevector(c0) - _mps.statevector(c1))) > 0.1


@needs_mps
class TestGPUExpectation:
    def test_bell_state(self):
        c = Circuit(2); c.h(0); c.cx(0, 1)
        assert _mps.expectation(c, Z(0) @ Z(1)) == pytest.approx(1.0, abs=1e-5)
        assert _mps.expectation(c, 1.0 * Z(0)) == pytest.approx(0.0, abs=1e-5)
        assert _mps.expectation(c, X(0) @ X(1)) == pytest.approx(1.0, abs=1e-5)

    def test_matches_dense_reference(self):
        c = _all_gates_circuit()
        H = Z(0) @ Z(1) + 0.5 * X(0) + 0.3 * Y(2) @ Z(3) + 0.7 * X(1) @ X(2)
        sv = cpu.statevector(c)
        ref = float(np.real(np.vdot(sv, H.to_matrix(4) @ sv)))
        assert _mps.expectation(c, H) == pytest.approx(ref, abs=1e-5)

    def test_matrix_free_fallback_matches_dense(self):
        c = _all_gates_circuit()
        H = Z(0) @ Z(1) + 0.5 * X(0) + 0.3 * Y(2) @ Z(3) + 0.7 * X(1) @ X(2)
        sv = cpu.statevector(c)
        ref = float(np.real(np.vdot(sv, H.to_matrix(4) @ sv)))
        fb = MPSBackend._expectation_from_statevector(sv, H, 4)
        assert fb == pytest.approx(ref, abs=1e-9)

    def test_large_qubit_count(self):
        # Dense-matrix path would need a 2^18 x 2^18 matrix (~1 TB); the GPU
        # path must handle this without materializing anything dense.
        n = 18
        c = Circuit(n)
        c.h(0)
        for q in range(n - 1):
            c.cx(q, q + 1)
        H = sum((Z(q) @ Z(q + 1) for q in range(1, n - 1)), Z(0) @ Z(1))
        # GHZ state: every <Z Z> term is +1
        assert _mps.expectation(c, H) == pytest.approx(n - 1, abs=1e-3)


@needs_mps
class TestAdjointGradient:
    def test_small_qubit_gradient(self):
        # Small qubit counts exercise the partial threadgroup path that the
        # broken reduction corrupted.
        from metalq.parameter import Parameter
        pa, pb, pg = Parameter('a'), Parameter('b'), Parameter('g')
        c = Circuit(3)
        c.rx(pa, 0); c.ry(pb, 1); c.cx(0, 1); c.rz(pg, 2); c.cx(1, 2)
        H = Z(0) @ Z(1) + 0.5 * X(2)
        th = [0.3, 0.8, 1.2]

        grad = _mps.gradient(c, H, th, method='adjoint')

        eps = 1e-4
        for i in range(3):
            tp, tm = list(th), list(th)
            tp[i] += eps; tm[i] -= eps
            ep = cpu.expectation(c, H, params=tp)
            em = cpu.expectation(c, H, params=tm)
            num = (ep - em) / (2 * eps)
            assert grad[i] == pytest.approx(num, abs=1e-3), f"param {i}"


@needs_mps
class TestMPSExtendedGateParity:
    """Gates fixed in the fail-closed / U3+CH pass: u, u2, u3, r, ch, cs, csdg.

    These were previously mapped to identity (type 999) and produced silently
    wrong statevectors on the GPU.
    """

    @pytest.mark.parametrize("apply_mps, apply_cpu", [
        # u / u3 / u2 / r have exact CPU equivalents.
        (lambda c: c.u(0.7, 0.3, 1.1, 1),  lambda c: c.u(0.7, 0.3, 1.1, 1)),
        (lambda c: c.u3(0.7, 0.3, 1.1, 2), lambda c: c.u3(0.7, 0.3, 1.1, 2)),
        (lambda c: c.u2(0.3, 1.1, 1),      lambda c: c.u2(0.3, 1.1, 1)),
        (lambda c: c.r(0.7, 1.1, 2),       lambda c: c.r(0.7, 1.1, 2)),
        (lambda c: c.ch(0, 2),             lambda c: c.ch(0, 2)),
        # cs / csdg are not in the CPU gate set; compare against the exact
        # controlled-phase equivalents CP(+/-pi/2).
        (lambda c: c.cs(0, 3),   lambda c: c.cp(np.pi / 2, 0, 3)),
        (lambda c: c.csdg(0, 3), lambda c: c.cp(-np.pi / 2, 0, 3)),
    ])
    def test_gate_matches_cpu(self, apply_mps, apply_cpu):
        cm = Circuit(4)
        cc = Circuit(4)
        for c in (cm, cc):
            for q in range(4):
                c.h(q)
            c.rz(0.3, 0); c.ry(0.5, 1); c.rz(0.7, 2); c.ry(0.9, 3)
        apply_mps(cm)
        apply_cpu(cc)
        np.testing.assert_allclose(
            _mps.statevector(cm), cpu.statevector(cc), atol=1e-5)

    def test_unsupported_gate_raises(self):
        from metalq.circuit import Gate
        from metalq.exceptions import ValidationError
        c = Circuit(2)
        c.h(0)
        # Genuinely unmappable gate name -> must fail closed, never identity.
        c._gates.append(Gate(name='bogus_gate', qubits=[0], params=[]))
        with pytest.raises(ValidationError):
            _mps.statevector(c)

    def test_unmapped_real_gate_raises(self):
        from metalq.exceptions import ValidationError
        # rxx is a real gate the CPU supports but the MPS backend does not map;
        # it must raise rather than silently apply identity.
        c = Circuit(2)
        c.rxx(0.3, 0, 1)
        with pytest.raises(ValidationError):
            _mps.statevector(c)

    def test_id_and_barrier_are_noops(self):
        cm = Circuit(2); cm.h(0); cm.id(1); cm.barrier(); cm.cx(0, 1)
        cc = Circuit(2); cc.h(0); cc.cx(0, 1)
        np.testing.assert_allclose(
            _mps.statevector(cm), cpu.statevector(cc), atol=1e-5)


@needs_mps
class TestAdjointGradientPCP:
    """Adjoint gradients for P and CP parameters (previously returned 0)."""

    def _check(self, build, params):
        c = build()
        H = Z(0) @ Z(1) + 0.5 * X(0)
        grad = _mps.gradient(c, H, params, method='adjoint')
        ref = cpu.gradient(c, H, params, method='parameter_shift')
        np.testing.assert_allclose(grad, ref, atol=1e-5)

    def test_p_gate_ansatz(self):
        from metalq.parameter import Parameter
        pa, pb, pc = Parameter('a'), Parameter('b'), Parameter('c')
        self._check(
            lambda: Circuit(2).h(0).ry(pa, 0).p(pb, 0).rx(pc, 1),
            [0.3, 0.9, 0.5])

    def test_cp_gate_ansatz(self):
        from metalq.parameter import Parameter
        pa, pb, pc = Parameter('a'), Parameter('b'), Parameter('c')
        self._check(
            lambda: Circuit(2).h(0).h(1).rx(pa, 0).cp(pb, 0, 1).ry(pc, 1),
            [0.4, 1.1, 0.7])

    def test_mixed_rx_p_cp(self):
        from metalq.parameter import Parameter
        p = [Parameter(f'p{i}') for i in range(4)]
        vals = [0.5, 1.0, 0.7, 1.3]
        self._check(
            lambda: (Circuit(3).h(0).h(1).rx(p[0], 0).p(p[1], 1)
                     .cp(p[2], 0, 1).rz(p[3], 2).cx(1, 2)),
            vals)

    def test_u3_parameterized_falls_back(self):
        # A parameterized u3 is not natively adjoint-differentiable; gradient()
        # must fall back to parameter-shift and still be correct.
        from metalq.parameter import Parameter
        th, ph, lm = Parameter('t'), Parameter('p'), Parameter('l')
        c = Circuit(2)
        c.h(0); c.u3(th, ph, lm, 0); c.cx(0, 1)
        H = Z(0) @ Z(1) + 0.5 * X(0)
        vals = [0.4, 0.9, 1.2]
        grad = _mps.gradient(c, H, vals, method='adjoint')
        ref = cpu.gradient(c, H, vals, method='parameter_shift')
        np.testing.assert_allclose(grad, ref, atol=1e-5)


@needs_mps
class TestHamiltonianMarshaling:
    def test_non_hermitian_coeff_raises(self):
        from metalq.exceptions import ValidationError
        c = Circuit(2); c.h(0)
        H = (0.5 + 0.3j) * Z(0)
        with pytest.raises(ValidationError):
            _mps.expectation(c, H)

    def test_tiny_imaginary_noise_ok(self):
        # Float noise below tolerance must not raise.
        c = Circuit(2); c.h(0)
        H = (1.0 + 1e-13j) * Z(0) @ Z(1)
        _mps.expectation(c, H)  # should not raise


@needs_mps
class TestFusedExpectationAndGradient:
    """metalq_gradient_adjoint_energy: energy + gradient in one GPU pass.

    Verifies the fused expectation_and_gradient() path agrees with the
    separate expectation()/gradient(method='adjoint') calls it is meant to
    replace, on both the native and fallback code paths.
    """

    def _mixed_ansatz(self, n=10):
        from metalq.parameter import Parameter
        params = [Parameter(f'p{i}') for i in range(3 * n)]
        c = Circuit(n)
        idx = 0
        for q in range(n):
            c.rx(params[idx], q); idx += 1
        for q in range(n - 1):
            c.cx(q, q + 1)
        for q in range(n):
            c.ry(params[idx], q); idx += 1
        for q in range(n - 1):
            c.cx(q, q + 1)
        for q in range(n):
            c.rz(params[idx], q); idx += 1
        vals = list(np.linspace(0.1, 2.0, len(params)))
        return c, vals

    def test_matches_separate_calls(self):
        c, vals = self._mixed_ansatz(10)
        H = Z(0) @ Z(1) + 0.5 * X(0)
        for q in range(2, 9):
            H = H + Z(q) @ Z(q + 1)

        energy, grad = _mps.expectation_and_gradient(c, H, vals)
        ref_energy = _mps.expectation(c, H, vals)
        ref_grad = _mps.gradient(c, H, vals, method='adjoint')

        assert energy == pytest.approx(ref_energy, abs=1e-6)
        np.testing.assert_allclose(grad, ref_grad, atol=1e-9)

    def test_hamiltonian_with_y_terms(self):
        c, vals = self._mixed_ansatz(10)
        H = Y(0) @ Y(1) + 0.3 * Z(2) + 0.5 * X(3) @ Y(4)

        energy, grad = _mps.expectation_and_gradient(c, H, vals)
        ref_energy = _mps.expectation(c, H, vals)
        ref_grad = _mps.gradient(c, H, vals, method='adjoint')

        assert energy == pytest.approx(ref_energy, abs=1e-6)
        np.testing.assert_allclose(grad, ref_grad, atol=1e-9)

    def test_u3_fallback_matches_cpu(self):
        # A parameterized u3 forces the parameter-shift fallback path inside
        # expectation_and_gradient(); result must still match the CPU
        # reference (expectation + parameter-shift gradient).
        from metalq.parameter import Parameter
        th, ph, lm = Parameter('t'), Parameter('p'), Parameter('l')
        c = Circuit(2)
        c.h(0); c.u3(th, ph, lm, 0); c.cx(0, 1)
        H = Z(0) @ Z(1) + 0.5 * X(0)
        vals = [0.4, 0.9, 1.2]

        energy, grad = _mps.expectation_and_gradient(c, H, vals)
        ref_energy = cpu.expectation(c, H, vals)
        ref_grad = cpu.gradient(c, H, vals, method='parameter_shift')

        assert energy == pytest.approx(ref_energy, abs=1e-5)
        np.testing.assert_allclose(grad, ref_grad, atol=1e-5)

    def test_zero_parameter_circuit(self):
        c = Circuit(2); c.h(0); c.cx(0, 1)
        H = Z(0) @ Z(1)

        energy, grad = _mps.expectation_and_gradient(c, H, [])

        assert energy == pytest.approx(1.0, abs=1e-5)
        assert len(grad) == 0


@needs_mps
class TestGPUSampling:
    """GPU-resident measurement sampling (metalq_sample): no full readback.

    The MPSBackend.run() counts path samples on the GPU and must (a) produce
    the SAME bitstring-key format as the CPU sampler, (b) reproduce the true
    Born-rule distribution, and (c) be deterministic under np.random.seed.
    """

    def test_native_sample_available(self):
        # These tests exercise the native path; if the loaded dylib predates
        # metalq_sample there is nothing to verify here.
        assert hasattr(_mps._lib, 'metalq_sample'), \
            "metalq_sample not exported; rebuild native (cd native && make)"

    def test_deterministic_outcome_and_format_parity(self):
        # X on qubit 1 of a 3-qubit register -> deterministic |010> = index 2.
        c = Circuit(3); c.x(1)
        gpu_counts = _mps.run(c, shots=64)['counts']
        cpu_counts = cpu.run(c, shots=64)['counts']

        assert len(gpu_counts) == 1
        (gpu_key,) = gpu_counts.keys()
        (cpu_key,) = cpu_counts.keys()
        # Exact key parity with the CPU sampler and the expected little-endian
        # bitstring (qubit 1 set -> '010').
        assert gpu_key == cpu_key == '010'
        assert gpu_counts[gpu_key] == 64

    def test_statevector_omitted_when_sampling(self):
        c = Circuit(2); c.h(0); c.cx(0, 1)
        res = _mps.run(c, shots=100)
        assert 'counts' in res
        assert 'statevector' not in res  # no readback when only counts wanted

    def test_shots_zero_still_returns_statevector(self):
        c = Circuit(2); c.h(0); c.cx(0, 1)
        res = _mps.run(c, shots=0)
        assert 'statevector' in res
        assert 'counts' not in res

    def test_bell_state_frequencies(self):
        c = Circuit(2); c.h(0); c.cx(0, 1)
        shots = 100_000
        counts = _mps.run(c, shots=shots)['counts']
        assert set(counts.keys()) == {'00', '11'}
        assert counts['00'] / shots == pytest.approx(0.5, abs=0.01)
        assert counts['11'] / shots == pytest.approx(0.5, abs=0.01)

    def test_random_circuit_tvd(self):
        # 3-qubit random circuit: sampled frequencies vs GPU statevector
        # probabilities, total-variation distance < 0.02 at 100k shots.
        np.random.seed(1234)
        c = Circuit(3)
        for q in range(3):
            c.ry(float(np.random.uniform(0, np.pi)), q)
        c.cx(0, 1); c.cx(1, 2)
        for q in range(3):
            c.rz(float(np.random.uniform(0, np.pi)), q)
            c.rx(float(np.random.uniform(0, np.pi)), q)

        sv = _mps.statevector(c)
        true_probs = np.abs(sv) ** 2
        true_probs = true_probs / true_probs.sum()

        shots = 100_000
        counts = _mps.run(c, shots=shots)['counts']
        emp = np.zeros(8)
        for key, cnt in counts.items():
            emp[int(key, 2)] = cnt / shots

        tvd = 0.5 * np.abs(emp - true_probs).sum()
        assert tvd < 0.02, f"TVD too large: {tvd}"

    def test_reproducibility(self):
        c = Circuit(3)
        c.h(0); c.h(1); c.ry(0.7, 2); c.cx(0, 2)

        np.random.seed(42)
        counts_a = _mps.run(c, shots=20_000)['counts']
        np.random.seed(42)
        counts_b = _mps.run(c, shots=20_000)['counts']
        assert counts_a == counts_b

        # A different seed should (almost surely) give a different sample set.
        np.random.seed(43)
        counts_c = _mps.run(c, shots=20_000)['counts']
        assert counts_a != counts_c
