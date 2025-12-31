"""
tests/test_phase1.py - Phase 1 Core API Tests
"""
import pytest
import numpy as np
from metalq import Circuit, Parameter, Result
from metalq.spin import X, Y, Z, I, PauliTerm, Hamiltonian


class TestParameter:
    def test_create_parameter(self):
        theta = Parameter('θ')
        assert theta.name == 'θ'
    
    def test_parameter_equality(self):
        p1 = Parameter('θ')
        p2 = Parameter('θ')  # Same name, different instance
        assert p1 != p2  # Different UUID
        assert p1 == p1
    
    def test_parameter_expression(self):
        theta = Parameter('θ')
        expr = 2 * theta + 0.5
        
        result = expr.evaluate({theta: 1.0})
        assert result == 2.5


class TestCircuit:
    def test_create_circuit(self):
        qc = Circuit(4)
        assert qc.num_qubits == 4
        assert qc.depth == 0
    
    def test_single_qubit_gates(self):
        qc = Circuit(1)
        qc.h(0).x(0).y(0).z(0)
        assert qc.size == 4
    
    def test_method_chaining(self):
        qc = Circuit(2).h(0).cx(0, 1).measure_all()
        assert qc.size == 2
        assert len(qc.measurements) == 2
    
    def test_parameterized_circuit(self):
        theta = Parameter('θ')
        qc = Circuit(1).ry(theta, 0)
        
        assert len(qc.parameters) == 1
        assert qc.parameters[0] == theta
    
    def test_bind_parameters(self):
        theta = Parameter('θ')
        qc = Circuit(1).ry(theta, 0)
        
        bound = qc.bind_parameters({theta: 0.5})
        assert len(bound.parameters) == 0
    
    def test_depth_calculation(self):
        qc = Circuit(3)
        qc.h(0).h(1).h(2)  # Depth 1 (parallel)
        qc.cx(0, 1)         # Depth 2
        qc.cx(1, 2)         # Depth 3
        assert qc.depth == 3
    
    def test_inverse(self):
        qc = Circuit(1).rx(0.5, 0).ry(0.3, 0)
        inv = qc.inverse()
        
        # inverse reverses order and negates angles
        assert len(inv.gates) == 2
        assert inv.gates[0].name == 'ry'
        assert inv.gates[0].params[0] == -0.3
    
    def test_compose(self):
        qc1 = Circuit(2).h(0)
        qc2 = Circuit(2).cx(0, 1)
        
        composed = qc1 + qc2
        assert composed.size == 2
    
    def test_draw_ascii(self):
        qc = Circuit(2).h(0).cx(0, 1)
        text = qc.draw('text')
        assert 'H' in text
    
    def test_draw_latex(self):
        qc = Circuit(2).h(0).cx(0, 1)
        latex = qc.draw('latex')
        assert 'quantikz' in latex


class TestResult:
    def test_counts(self):
        result = Result(counts={'00': 500, '11': 500})
        assert result.total_counts == 1000
        assert result.probability('00') == 0.5
    
    def test_most_frequent(self):
        result = Result(counts={'00': 600, '11': 400})
        assert result.most_frequent() == '00'
    
    def test_statevector(self):
        sv = np.array([1, 0, 0, 0], dtype=np.complex128)
        result = Result(statevector=sv)
        assert result.statevector is not None


class TestSpin:
    def test_pauli_operators(self):
        x = X(0)
        assert x.ops == [('X', 0)]
        assert x.coeff == 1.0
    
    def test_tensor_product(self):
        zz = Z(0) @ Z(1)
        assert len(zz.ops) == 2
    
    def test_scalar_multiplication(self):
        term = -1.0 * Z(0)
        assert term.coeff == -1.0
    
    def test_hamiltonian_addition(self):
        H = Z(0) + Z(1)
        assert isinstance(H, Hamiltonian)
        assert len(H.terms) == 2
    
    def test_complex_hamiltonian(self):
        H = -1.0 * Z(0) @ Z(1) + 0.5 * X(0) + 0.5 * X(1)
        assert len(H.terms) == 3
    
    def test_to_matrix(self):
        H = Z(0)
        matrix = H.to_matrix(1)
        expected = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        np.testing.assert_array_almost_equal(matrix, expected)
