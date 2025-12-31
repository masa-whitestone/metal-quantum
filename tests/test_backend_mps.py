"""
tests/test_backend_mps.py - MPS Integration Tests
"""
import pytest
import numpy as np
from metalq import Circuit, Parameter
from metalq.backends.mps.backend import MPSBackend

class TestMPSBackend:
    def setup_method(self):
        try:
            self.backend = MPSBackend()
        except RuntimeError as e:
            pytest.skip(f"MPS Backend not available: {e}")
            
        if not self.backend.is_available():
            pytest.skip("Metal not supported on this device")
    
    def test_single_qubit_gates(self):
        # X gate: |0> -> |1>
        qc = Circuit(1).x(0)
        sv = self.backend.statevector(qc)
        assert np.isclose(np.abs(sv[0]), 0)
        assert np.isclose(np.abs(sv[1]), 1)
        
        # H gate
        qc = Circuit(1).h(0)
        sv = self.backend.statevector(qc)
        assert np.isclose(sv[0], 1/np.sqrt(2))
        assert np.isclose(sv[1], 1/np.sqrt(2))

    def test_two_qubit_gates(self):
        # Bell state
        qc = Circuit(2).h(0).cx(0, 1)
        sv = self.backend.statevector(qc)
        
        # |00> + |11>
        assert np.isclose(sv[0], 1/np.sqrt(2))
        assert np.isclose(sv[3], 1/np.sqrt(2))
        assert np.abs(sv[1]) < 1e-6
        assert np.abs(sv[2]) < 1e-6
    
    def test_parameter_binding(self):
        theta = Parameter('θ')
        qc = Circuit(1).rx(theta, 0)
        # RX(π) ~ -iX
        # |0> -> -i|1>
        res = self.backend.run(qc, params={theta: np.pi}, shots=0)
        sv = res['statevector']
        assert np.abs(sv[0]) < 1e-6
        assert np.isclose(np.abs(sv[1]), 1)
