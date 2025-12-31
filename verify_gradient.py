
import numpy as np
from metalq import Circuit, Parameter
from metalq.spin import Z
from metalq.backends.mps.backend import MPSBackend

def test_gradient():
    print("Testing Adjoint Differentiation...")
    
    theta = Parameter('theta')
    qc = Circuit(1)
    qc.rx(theta, 0)
    
    H = Z(0)
    
    backend = MPSBackend()
    
    # 1. Parameter Shift (Reference)
    # The derivative of <Z> for RX(theta) |0> is -sin(theta)
    # At theta = pi/2: -1.0
    
    params = [np.pi/2]
    
    print("Running Parameter Shift...")
    grad_ps = backend.gradient(qc, H, params, method='parameter_shift')
    print(f"PS Gradient: {grad_ps}")
    
    # 2. Adjoint
    print("Running Adjoint...")
    grad_adj = backend.gradient(qc, H, params, method='adjoint')
    print(f"Adjoint Gradient: {grad_adj}")
    
    diff = np.abs(grad_ps - grad_adj)
    print(f"Difference: {diff}")
    
    if np.all(diff < 1e-4):
        print("SUCCESS: Adjoint matches Parameter Shift")
    else:
        print("FAILURE: Gradients mismatch")

if __name__ == "__main__":
    test_gradient()
