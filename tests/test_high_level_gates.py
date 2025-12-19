"""
test_high_level_gates.py - High-level gate support verification
"""
import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    QFTGate, QFT, 
    MCXGate, MCPhaseGate,
    GroverOperator,
    UnitaryGate,
    DiagonalGate,
    PermutationGate
)
from qiskit.quantum_info import Statevector
import metalq


class TestQFT:
    """QFT related tests"""
    
    def test_qft_circuit(self):
        """QFT (QuantumCircuit version) - Pre-unrolled"""
        qc = QuantumCircuit(3)
        qc.x(0)  # |001>
        qc.compose(QFT(3), inplace=True)
        
        sv_qiskit = Statevector(qc).data
        sv_metalq = metalq.statevector(qc)
        
        np.testing.assert_array_almost_equal(
            np.abs(sv_qiskit), np.abs(sv_metalq), decimal=4
        )
    
    def test_qftgate(self):
        """QFTGate (Gate version) - Needs decomposition"""
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.append(QFTGate(3), [0, 1, 2])
        
        sv_qiskit = Statevector(qc).data
        sv_metalq = metalq.statevector(qc)
        
        np.testing.assert_array_almost_equal(
            np.abs(sv_qiskit), np.abs(sv_metalq), decimal=4
        )
    
    def test_qft_inverse(self):
        """QFT + QFT_dagger = Identity"""
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.x(2)  # |101>
        qc.append(QFTGate(3), [0, 1, 2])
        qc.append(QFTGate(3).inverse(), [0, 1, 2])
        
        sv = metalq.statevector(qc)
        # |101> = index 5
        expected = np.zeros(8, dtype=complex)
        expected[5] = 1
        
        np.testing.assert_array_almost_equal(np.abs(sv), np.abs(expected), decimal=4)


class TestMCX:
    """Multi-Controlled X Gates"""
    
    def test_mcx_3_controls(self):
        """MCX with 3 controls = C3X"""
        qc = QuantumCircuit(4)
        qc.x([0, 1, 2])  # All controls = 1
        qc.append(MCXGate(3), [0, 1, 2, 3])
        
        sv_qiskit = Statevector(qc).data
        sv_metalq = metalq.statevector(qc)
        
        # Compare probabilities to ignore Global Phase accumulation errors in float32
        prob_qiskit = np.abs(sv_qiskit) ** 2
        prob_metalq = np.abs(sv_metalq) ** 2
        np.testing.assert_array_almost_equal(prob_qiskit, prob_metalq, decimal=3)
    
    def test_mcx_4_controls(self):
        """MCX with 4 controls"""
        qc = QuantumCircuit(5)
        qc.x([0, 1, 2, 3])  # All controls = 1
        qc.append(MCXGate(4), [0, 1, 2, 3, 4])
        
        sv_qiskit = Statevector(qc).data
        sv_metalq = metalq.statevector(qc)
        
        prob_qiskit = np.abs(sv_qiskit) ** 2
        prob_metalq = np.abs(sv_metalq) ** 2
        np.testing.assert_array_almost_equal(prob_qiskit, prob_metalq, decimal=3)
    
    def test_mcx_not_triggered(self):
        """MCX should not flip if any control is 0"""
        qc = QuantumCircuit(4)
        qc.x([0, 2])  # Only 2 of 3 controls = 1
        qc.append(MCXGate(3), [0, 1, 2, 3])
        
        sv = metalq.statevector(qc)
        # Target should NOT flip, state should be |0101> = index 5
        expected = np.zeros(16, dtype=complex)
        expected[5] = 1
        
        np.testing.assert_array_almost_equal(np.abs(sv), np.abs(expected), decimal=4)


class TestGrover:
    """Grover's algorithm components"""
    
    def test_grover_diffuser(self):
        """Grover diffuser (amplitude amplification)"""
        n = 3
        qc = QuantumCircuit(n)
        
        # Superposition
        qc.h(range(n))
        
        # Mark |111>
        qc.x(range(n))
        qc.h(n-1)
        qc.mcx(list(range(n-1)), n-1)
        qc.h(n-1)
        qc.x(range(n))
        
        # Diffuser
        qc.h(range(n))
        qc.x(range(n))
        qc.h(n-1)
        qc.mcx(list(range(n-1)), n-1)
        qc.h(n-1)
        qc.x(range(n))
        qc.h(range(n))
        
        sv_qiskit = Statevector(qc).data
        sv_metalq = metalq.statevector(qc)
        
        prob_qiskit = np.abs(sv_qiskit) ** 2
        prob_metalq = np.abs(sv_metalq) ** 2
        np.testing.assert_array_almost_equal(prob_qiskit, prob_metalq, decimal=3)


class TestUnitaryGate:
    """Custom Unitary Gates"""
    
    def test_1q_unitary(self):
        """1Q Custom Unitary"""
        from scipy.stats import unitary_group
        U = unitary_group.rvs(2, random_state=42)
        
        qc = QuantumCircuit(1)
        qc.append(UnitaryGate(U), [0])
        
        sv_qiskit = Statevector(qc).data
        sv_metalq = metalq.statevector(qc)
        
        np.testing.assert_array_almost_equal(np.abs(sv_qiskit), np.abs(sv_metalq), decimal=4)
    
    def test_2q_unitary(self):
        """2Q Custom Unitary"""
        from scipy.stats import unitary_group
        U = unitary_group.rvs(4, random_state=42)
        
        qc = QuantumCircuit(2)
        qc.append(UnitaryGate(U), [0, 1])
        
        sv_qiskit = Statevector(qc).data
        sv_metalq = metalq.statevector(qc)
        
        np.testing.assert_array_almost_equal(np.abs(sv_qiskit), np.abs(sv_metalq), decimal=4)
    
    def test_known_unitary(self):
        """Known Unitary (CNOT)"""
        cnot = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        
        qc = QuantumCircuit(2)
        qc.x(1)  # |10> (q1=1, q0=0)
        qc.append(UnitaryGate(cnot), [0, 1])
        
        sv = metalq.statevector(qc)
        expected = np.array([0, 0, 0, 1], dtype=complex)  # |11>
        
        np.testing.assert_array_almost_equal(np.abs(sv), np.abs(expected), decimal=5)


class TestDiagonalGate:
    """DiagonalGate tests"""
    
    def test_diagonal_2q(self):
        """2Q Diagonal Gate"""
        diag = [1, 1j, -1, -1j]
        
        qc = QuantumCircuit(2)
        qc.h([0, 1])
        qc.append(DiagonalGate(diag), [0, 1])
        
        sv_qiskit = Statevector(qc).data
        sv_metalq = metalq.statevector(qc)
        
        np.testing.assert_array_almost_equal(np.abs(sv_qiskit), np.abs(sv_metalq), decimal=4)


class TestPermutationGate:
    """PermutationGate tests"""
    
    def test_permutation_swap(self):
        """Permutation [1, 0] = SWAP"""
        qc = QuantumCircuit(2)
        qc.x(0)  # |01>
        qc.append(PermutationGate([1, 0]), [0, 1])  # |10>
        
        sv = metalq.statevector(qc)
        expected = np.array([0, 0, 1, 0], dtype=complex)
        
        np.testing.assert_array_almost_equal(np.abs(sv), np.abs(expected), decimal=5)
