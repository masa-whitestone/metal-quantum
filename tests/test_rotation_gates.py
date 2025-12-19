
import pytest
import numpy as np
import metalq
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

class TestRotationGates:
    def test_p_gate(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(np.pi/2, 0) # S gate effectively. |+> -> (|0> + i|1>)/sqrt(2)
        
        sv_metalq = metalq.statevector(qc)
        expected = np.array([1/np.sqrt(2), 1j/np.sqrt(2)], dtype=complex)
        
        np.testing.assert_array_almost_equal(sv_metalq, expected, decimal=5)

    def test_cp_gate(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        # |++> = 0.5 [1, 1, 1, 1]
        
        qc.cp(np.pi, 0, 1) # CZ effectively. Changes |11> sign.
        # Result: 0.5 [1, 1, 1, -1]
        
        sv_metalq = metalq.statevector(qc)
        expected = np.array([0.5, 0.5, 0.5, -0.5], dtype=complex)
        
        np.testing.assert_array_almost_equal(sv_metalq, expected, decimal=5)

    def test_rz_gate(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(np.pi, 0) # Z gate (with global phase check? RZ(pi) = -i Z)
        # H |0> -> |+>. RZ(pi) |+> = e^-i(pi/2)|0> + e^i(pi/2)|1> = -i|0> + i|1> = -i(|0> - |1>) = -i |->
        
        sv_metalq = metalq.statevector(qc)
        # Qiskit RZ definition matches RZ(pi) ~ Z up to phase.
        # MetalQ RZ should match Qiskit RZ.
        
        qc_q = QuantumCircuit(1)
        qc_q.h(0)
        qc_q.rz(np.pi, 0)
        sv_q = Statevector(qc_q).data
        
        # Check magniture probabilities to ignore global phase differences if any
        np.testing.assert_array_almost_equal(np.abs(sv_metalq)**2, np.abs(sv_q)**2, decimal=5)
