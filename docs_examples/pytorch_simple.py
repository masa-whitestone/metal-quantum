import torch
import torch.optim as optim
from metalq import Circuit, Parameter, Hamiltonian, Z
from metalq.torch import QuantumLayer

print("=== PyTorch Integration: Simple Optimization ===")

# Define Hamiltonian: H = Z0
H = Z(0)

# Define Circuit with 1 parameter
qc = Circuit(1)
theta = Parameter('theta')
qc.rx(theta, 0)

# Create Layer
model = QuantumLayer(qc, H, backend_name='mps')
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Initial parameter
print(f"Initial param: {list(model.parameters())[0].item():.4f}")

# Optimization
print("\nStarting optimization (Target: Minimize <Z0> -> |1> state)...")
for i in range(21):
    optimizer.zero_grad()
    loss = model()
    loss.backward()
    optimizer.step()
    
    if i % 5 == 0:
        print(f"Step {i}: Loss = {loss.item():.4f}, Param = {list(model.parameters())[0].item():.4f}")

print("\nOptimization complete.")
