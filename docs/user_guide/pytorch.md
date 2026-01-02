# PyTorch Integration

Metal-Q uniquely integrates with PyTorch, allowing quantum circuits to be used as differentiable layers within neural networks (`nn.Module`). This enables hybrid quantum-classical machine learning (QML) on GPUs.

## Quantum Layer

The `QuantumLayer` class wraps a parameterized `Circuit` and an observable (`Hamiltonian`) into a PyTorch module. It automatically computes gradients using the Adjoint Differentiation method (or Parameter Shift rule on CPU).

### Example: Simple Optimization

In this example, we optimize a single parameter to minimize the expectation value of an observable.

```python
import torch
import torch.optim as optim
from metalq import Circuit, Parameter, Z
from metalq.torch import QuantumLayer

print("=== PyTorch Integration: Simple Optimization ===")

# 1. Define Hamiltonian: H = Z0
H = Z(0)

# 2. Define Circuit with 1 parameter
qc = Circuit(1)
theta = Parameter('theta')
qc.rx(theta, 0)

# 3. Create Quantum Layer
#    Input: None (handled internally or via batch inputs)
#    Output: Expectation value <Z0>
model = QuantumLayer(qc, H, backend_name='mps')
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Initial parameter
print(f"Initial param: {list(model.parameters())[0].item():.4f}")

# 4. Optimization Loop
print("\nStarting optimization (Target: Minimize <Z0> -> |1> state)...")
for i in range(21):
    optimizer.zero_grad()
    loss = model()     # Forward pass
    loss.backward()    # Backward pass (Adjoint Differentiation)
    optimizer.step()
    
    if i % 5 == 0:
        print(f"Step {i}: Loss = {loss.item():.4f}, Param = {list(model.parameters())[0].item():.4f}")

print("\nOptimization complete.")
```

### Output
```text
=== PyTorch Integration: Simple Optimization ===
Initial param: 3.8440

Starting optimization (Target: Minimize <Z0> -> |1> state)...
Step 0: Loss = -0.7633, Param = 3.7440
Step 5: Loss = -0.9770, Param = 3.2677
Step 10: Loss = -0.9887, Param = 2.9500
Step 15: Loss = -0.9720, Param = 2.9180
Step 20: Loss = -0.9937, Param = 3.0641

Optimization complete.
```

## Creating Hybrid Models

You can combine `QuantumLayer` with standard PyTorch layers (`nn.Linear`, `nn.Conv2d`, etc.) to build hybrid models.

```python
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classical_pre = nn.Linear(4, 2)
        self.quantum = QuantumLayer(qc, H)
        self.classical_post = nn.Linear(1, 1)
        
    def forward(self, x):
        x = self.classical_pre(x)
        x = torch.sigmoid(x)
        # Note: QuantumLayer currently might need adaptation for batch inputs
        # typical usage involves mapping classical features to rotation angles
        x_q = self.quantum(inputs={theta: x}) 
        return self.classical_post(x_q)
```
