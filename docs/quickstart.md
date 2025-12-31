# Quick Start

## 1. Basic Circuit Simulation

Running a simple Bell State circuit using Metal-Q's native API:

```python
from metalq import Circuit, run

# Create a circuit with 2 qubits
qc = Circuit(2)
qc.h(0)
qc.cx(0, 1)

# Run on MPS (Metal Performance Shaders) backend
result = run(qc, shots=1000, backend='mps')

print(f"Counts: {result.counts}")
# Counts: {'00': 502, '11': 498}
```

## 2. Variational Quantum Eigensolver (VQE) with PyTorch

Metal-Q integrates with PyTorch to optimize variational circuits efficiently.

```python
import torch
import torch.optim as optim
from metalq import Circuit, Parameter, Hamiltonian, Z, X
from metalq.torch import QuantumLayer

# Define Hamiltonian: H = Z0 * Z1
H = Z(0) * Z(1)

# Define Ansatz
circuit = Circuit(2)
theta = Parameter('theta')
circuit.rx(theta, 0)
circuit.cx(0, 1)

# Create PyTorch Layer
model = QuantumLayer(circuit, H, backend_name='mps')
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Optimization Loop
for step in range(100):
    optimizer.zero_grad()
    loss = model() # Expectation value
    loss.backward() # Computes gradients via GPU Adjoint Differentiation
    optimizer.step()
    
    if step % 20 == 0:
        print(f"Step {step}, Energy: {loss.item():.4f}")
```

## 3. Qiskit Interoperability

You can create circuits in Qiskit and simulate them on Metal-Q's high-performance backend.

```python
from qiskit import QuantumCircuit
from metalq.adapters.qiskit_adapter import to_metalq, to_qiskit
from metalq import run

# Qiskit Circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Convert to Metal-Q
mq_circuit = to_metalq(qc)

# Run on GPU
result = run(mq_circuit, shots=1000)
print(result.counts)

# Convert back to Qiskit (if needed)
qc_back = to_qiskit(mq_circuit)
```
