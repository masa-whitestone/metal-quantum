"""
tests/test_backend_cpu.py - CPU Backend Tests
"""
import pytest
import numpy as np
import time
from metalq import Circuit, Parameter, Result
from metalq.spin import Z, X
from metalq.backends.cpu.backend import CPUBackend
from metalq.backends.cpu.statevector import initialize_statevector

class TestCPUBackend:
    def setup_method(self):
        self.backend = CPUBackend()
    
    def test_bell_state(self):
        # |00> -> (|00> + |11>) / sqrt(2)
        qc = Circuit(2).h(0).cx(0, 1)
        
        result = self.backend.run(qc)
        sv = result['statevector']
        
        # Check amplitudes
        assert np.isclose(sv[0], 1/np.sqrt(2))  # |00>
        assert np.isclose(sv[3], 1/np.sqrt(2))  # |11>
        assert np.isclose(sv[1], 0)             # |01>
        assert np.isclose(sv[2], 0)             # |10>
    
    def test_shots_measurement(self):
        # H gate -> 50/50 0/1
        qc = Circuit(1).h(0)
        
        result = self.backend.run(qc, shots=1000)
        counts = result['counts']
        
        assert '0' in counts
        assert '1' in counts
        assert 400 < counts['0'] < 600
        assert 400 < counts['1'] < 600
    
    def test_parameter_binding(self):
        theta = Parameter('θ')
        qc = Circuit(1).rx(theta, 0)
        
        # RX(π) |0> = -i|1>
        result = self.backend.run(qc, params={theta: np.pi})
        sv = result['statevector']
        
        assert np.isclose(np.abs(sv[0]), 0)
        assert np.isclose(np.abs(sv[1]), 1)
    
    def test_expectation_value(self):
        # |+> state
        qc = Circuit(1).h(0)
        H = Z(0)
        
        # <+|Z|+> = 0
        val = self.backend.expectation(qc, H)
        assert np.isclose(val, 0, atol=1e-10)
        
        # <0|Z|0> = 1
        qc2 = Circuit(1)
        val2 = self.backend.expectation(qc2, H)
        assert np.isclose(val2, 1, atol=1e-10)
    
    def test_gradient(self):
        # <0|RX(θ)dagger Z RX(θ)|0> = cos(θ)
        # Analytical gradient: -sin(θ)
        theta = Parameter('θ')
        qc = Circuit(1).rx(theta, 0)
        H = Z(0)
        
        params = [np.pi/4]  # θ = 45 deg
        grad = self.backend.gradient(qc, H, params)
        
        expected = -np.sin(np.pi/4)
        assert np.isclose(grad[0], expected, rtol=1e-3)
    
    def test_ghz_state_numba(self):
        # 5-qubit GHZ
        n = 5
        qc = Circuit(n).h(0)
        for i in range(n-1):
            qc.cx(i, i+1)
            
        result = self.backend.run(qc)
        sv = result['statevector']
        
        assert np.isclose(sv[0], 1/np.sqrt(2))      # |0...0>
        assert np.isclose(sv[-1], 1/np.sqrt(2))     # |1...1>
