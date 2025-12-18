"""
test_basic.py - Basic verification
"""
import pytest
import numpy as np
from qiskit import QuantumCircuit
import metalq

class TestBasicGates:
    """Tests for basic single qubit gates"""
    
    def test_identity(self):
        qc = QuantumCircuit(1)
        qc.id(0)
        
        sv = metalq.statevector(qc)
        expected = np.array([1, 0], dtype=complex)
        np.testing.assert_array_almost_equal(sv, expected)
    
    def test_x_gate(self):
        qc = QuantumCircuit(1)
        qc.x(0)
        
        sv = metalq.statevector(qc)
        expected = np.array([0, 1], dtype=complex)
        np.testing.assert_array_almost_equal(sv, expected)
    
    def test_hadamard(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        
        sv = metalq.statevector(qc)
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        np.testing.assert_array_almost_equal(sv, expected, decimal=5)
    
    def test_cnot(self):
        qc = QuantumCircuit(2)
        qc.x(0)  # |10> (q0 is lower bit) -> NO, Qiskit is little endian for bits?
        # Qiskit qubit order: q0 is at index 0 (rightmost in bitstring if little endian)
        # But statevector is usually tensor product q_n ... q0
        # Let's check: q0=1 => |1>
        
        # MetalQ implementation treats q0 as LSB for indexing (bit 0 in integer index)
        # So |1> at q0 corresponds to index 1.
        
        qc.cx(0, 1)  # Control 0, Target 1. If 0 is 1, flip 1.
        
        # Initial: |00>
        # X(0) -> |01> (q1=0, q0=1) -> Index 1
        # CX(0, 1) -> Control q0=1, Target q1 flips -> |11> -> Index 3
        
        sv = metalq.statevector(qc)
        expected = np.array([0, 0, 0, 1], dtype=complex) # Index 3 is 1
        np.testing.assert_array_almost_equal(sv, expected)

class TestBellState:
    """Bell states"""
    
    def test_bell_state_statevector(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        
        sv = metalq.statevector(qc)
        # |00> + |11>
        expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        np.testing.assert_array_almost_equal(sv, expected, decimal=5)
    
    def test_bell_state_measurement(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        
        result = metalq.run(qc, shots=1000)
        counts = result.get_counts()
        
        # Should see mostly '00' and '11'
        print(f"Counts: {counts}")
        assert '00' in counts
        assert '11' in counts
        
        total = sum(counts.values())
        p00 = counts.get('00', 0) / total
        p11 = counts.get('11', 0) / total
        
        assert 0.4 < p00 < 0.6
        assert 0.4 < p11 < 0.6
