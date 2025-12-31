
import torch
import torch.optim as optim
import numpy as np
from metalq import Circuit, Parameter, Z
from metalq.torch.layer import QuantumLayer

def test_optimization():
    print("Testing PyTorch Optimization with Adjoint Differentiation...")
    
    # 1. Define Circuit: RX(theta)
    theta = Parameter('theta')
    qc = Circuit(1)
    qc.rx(theta, 0)
    
    # 2. Hamiltonian: Z
    H = Z(0)
    
    # 3. Create Layer
    # Use MPS backend to trigger Adjoint Diff
    layer = QuantumLayer(qc, H, backend_name='mps')
    
    # Initialize parameter to pi/2 (where Expectation = <+|Z|+> = 0)
    # We want to minimize <Z>. Target state |1> (theta=pi). Expectation -1.
    # Current init in layer is 0.1 ~ 0.
    # Let's set it to valid start.
    with torch.no_grad():
        layer.weights[0] = 0.5 * 3.14159 # Start at pi/2
        
    print(f"Initial param: {layer.weights.item()}, Expectation: {layer().item()}")
    
    # 4. Optimizer
    optimizer = optim.Adam(layer.parameters(), lr=0.1)
    
    # 5. Optimization Loop
    for i in range(50):
        optimizer.zero_grad()
        loss = layer() # We minimize Expectation directly
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Step {i}: Loss={loss.item()}, Param={layer.weights.item()}")

    print(f"Final Step: Loss={loss.item()}, Param={layer.weights.item()}")
    
    # Check result
    # Target: -1.0
    # Param should be ~ pi (3.14)
    if loss.item() < -0.99:
        print("SUCCESS: Optimized to ground state.")
    else:
        print("FAILURE: Did not converge.")

if __name__ == "__main__":
    test_optimization()
