"""
test_all_gates.py - Test all standard gates
"""
import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import metalq


class Test1QGates:
    """Test all 1-qubit gates"""
    
    @pytest.mark.parametrize("gate_name,gate_method", [
        ("id", lambda qc: qc.id(0)),
        ("x", lambda qc: qc.x(0)),
        ("y", lambda qc: qc.y(0)),
        ("z", lambda qc: qc.z(0)),
        ("h", lambda qc: qc.h(0)),
        ("s", lambda qc: qc.s(0)),
        ("sdg", lambda qc: qc.sdg(0)),
        ("t", lambda qc: qc.t(0)),
        ("tdg", lambda qc: qc.tdg(0)),
        ("sx", lambda qc: qc.sx(0)),
        ("sxdg", lambda qc: qc.sxdg(0)),
    ])
    def test_gate_from_zero(self, gate_name, gate_method):
        """Test gate applied to |0>"""
        qc = QuantumCircuit(1)
        gate_method(qc)
        
        # Qiskit 2.x standard: global phase might differ, so we compare density matrices or abs
        sv_qiskit = Statevector(qc).data
        sv_metalq = metalq.statevector(qc)
        
        # Compare abs (ignores global phase)
        np.testing.assert_array_almost_equal(
            np.abs(sv_qiskit), np.abs(sv_metalq), decimal=5,
            err_msg=f"Gate {gate_name} failed on |0>"
        )
    
    @pytest.mark.parametrize("gate_name,gate_method", [
        ("id", lambda qc: qc.id(0)),
        ("x", lambda qc: qc.x(0)),
        ("y", lambda qc: qc.y(0)),
        ("z", lambda qc: qc.z(0)),
        ("h", lambda qc: qc.h(0)),
        ("s", lambda qc: qc.s(0)),
        ("sdg", lambda qc: qc.sdg(0)),
        ("t", lambda qc: qc.t(0)),
        ("tdg", lambda qc: qc.tdg(0)),
        ("sx", lambda qc: qc.sx(0)),
        ("sxdg", lambda qc: qc.sxdg(0)),
    ])
    def test_gate_from_one(self, gate_name, gate_method):
        """Test gate applied to |1>"""
        qc = QuantumCircuit(1)
        qc.x(0)  # |1>
        gate_method(qc)
        
        sv_qiskit = Statevector(qc).data
        sv_metalq = metalq.statevector(qc)
        
        np.testing.assert_array_almost_equal(
            np.abs(sv_qiskit), np.abs(sv_metalq), decimal=5,
            err_msg=f"Gate {gate_name} failed on |1>"
        )


class TestMCXDecomposition:
    """Test multi-controlled gates via decomposition"""
    
    def test_mcx_4_controls(self):
        """MCX with 4 controls should decompose correctly"""
        from qiskit.circuit.library import MCXGate
        
        qc = QuantumCircuit(5)
        for i in range(4):
            qc.x(i)  # All controls = 1
        qc.append(MCXGate(4), [0, 1, 2, 3, 4])
        
        sv_qiskit = Statevector(qc).data
        sv_metalq = metalq.statevector(qc)
        np.testing.assert_array_almost_equal(np.abs(sv_qiskit), np.abs(sv_metalq), decimal=4)

class TestQFT:
    """Test Quantum Fourier Transform"""
    
    def test_qft_3_qubits(self):
        """QFT on 3 qubits"""
        from qiskit.circuit.library import QFT
        
        qc = QuantumCircuit(3)
        qc.x(0)  # |001>
        qc.append(QFT(3), [0, 1, 2])
        
        sv_qiskit = Statevector(qc).data
        sv_metalq = metalq.statevector(qc)
        np.testing.assert_array_almost_equal(np.abs(sv_qiskit), np.abs(sv_metalq), decimal=4)

class TestCustomUnitary:
    """Test UnitaryGate logic"""
    
    def test_custom_1q(self):
        from qiskit.circuit.library import UnitaryGate
        # Define random unitary or known one (Hadamard)
        mat = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        gate = UnitaryGate(mat)
        
        qc = QuantumCircuit(1)
        qc.append(gate, [0])
        
        sv_qiskit = Statevector(qc).data
        sv_metalq = metalq.statevector(qc)
        np.testing.assert_array_almost_equal(np.abs(sv_qiskit), np.abs(sv_metalq), decimal=5)

    def test_custom_2q(self):
        from qiskit.circuit.library import UnitaryGate
        # CNOT matrix
        mat = np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,0,1],
            [0,0,1,0]
        ])
        gate = UnitaryGate(mat)
        
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.append(gate, [0, 1])
        
        sv_qiskit = Statevector(qc).data
        sv_metalq = metalq.statevector(qc)
        np.testing.assert_array_almost_equal(np.abs(sv_qiskit), np.abs(sv_metalq), decimal=5)
    

class TestRotationGates:
    """Test rotation gates with various angles"""
    
    @pytest.mark.parametrize("angle", [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2])
    def test_rx(self, angle):
        qc = QuantumCircuit(1)
        qc.rx(angle, 0)
        
        sv_qiskit = Statevector(qc).data
        sv_metalq = metalq.statevector(qc)
        np.testing.assert_array_almost_equal(np.abs(sv_qiskit), np.abs(sv_metalq), decimal=5)
    
    @pytest.mark.parametrize("angle", [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2])
    def test_ry(self, angle):
        qc = QuantumCircuit(1)
        qc.ry(angle, 0)
        
        sv_qiskit = Statevector(qc).data
        sv_metalq = metalq.statevector(qc)
        np.testing.assert_array_almost_equal(np.abs(sv_qiskit), np.abs(sv_metalq), decimal=5)
    
    @pytest.mark.parametrize("angle", [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2])
    def test_rz(self, angle):
        qc = QuantumCircuit(1)
        qc.h(0)  # |+> state
        qc.rz(angle, 0)
        
        sv_qiskit = Statevector(qc).data
        sv_metalq = metalq.statevector(qc)
        np.testing.assert_array_almost_equal(np.abs(sv_qiskit), np.abs(sv_metalq), decimal=5)
    
    @pytest.mark.parametrize("angle", [0, np.pi/4, np.pi/2, np.pi])
    def test_u_gate(self, angle):
        """Test generic U gate"""
        qc = QuantumCircuit(1)
        # U(theta, phi, lambda)
        qc.u(angle, angle/2, angle/3, 0)
        
        sv_qiskit = Statevector(qc).data
        sv_metalq = metalq.statevector(qc)
        np.testing.assert_array_almost_equal(np.abs(sv_qiskit), np.abs(sv_metalq), decimal=5)
