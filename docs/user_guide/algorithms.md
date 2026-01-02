# Algorithms

Metal-Q provides built-in implementations of popular quantum algorithms, leveraging the performance of the Metal backend.

## VQE (Variational Quantum Eigensolver)

VQE is a hybrid algorithm used to find the ground state energy of a Hamiltonian.

### Example: H2 Molecule

Here is a simplified example of finding the ground state energy for a Hydrogen molecule.

```python
from metalq import Circuit, Parameter, Z, X, Hamiltonian
from metalq.algorithms import VQE
import numpy as np

print("=== VQE: H2 Molecule (Simplified) ===")

# 1. Define Hamiltonian
# H2 simplified 2-qubit mapping
H = 0.39 * Z(0) + 0.39 * Z(1) + 0.18 * (Z(0) * Z(1)) + 0.01 * (X(0) * X(1))
print(f"Hamiltonian terms: {len(H.terms)}")

# 2. Construct Ansatz (Hardware Efficient)
qc = Circuit(2)
theta = [Parameter(f't{i}') for i in range(4)]

qc.ry(theta[0], 0)
qc.ry(theta[1], 1)
qc.cx(0, 1)
qc.ry(theta[2], 0)
qc.ry(theta[3], 1)

print("Ansatz constructed.")

# 3. Run VQE
# VQE init takes ansatz and optimizer options
vqe = VQE(qc, optimizer_kwargs={'lr': 0.1})

print("\nRunning VQE...")
# compute_minimum_eigenvalue takes hamiltonian
result = vqe.compute_minimum_eigenvalue(H, max_iter=50)

print(f"Minimum Eigenvalue found: {result.eigenvalue:.6f}")
print(f"Optimal Parameters: {[f'{p:.4f}' for p in result.optimal_params]}")
print(f"Total offset (approx -1.05): {result.eigenvalue - 1.05:.6f} Ha")
```

### Output
```text
=== VQE: H2 Molecule (Simplified) ===
Hamiltonian terms: 4
Ansatz constructed.

Running VQE...
Minimum Eigenvalue found: -0.593803
Optimal Parameters: ['-0.1563', '-2.0601', '3.2037', '0.4953']
Total offset (approx -1.05): -1.643803 Ha
```

## QAOA (Quantum Approximate Optimization Algorithm)

QAOA is used for solving combinatorial optimization problems.

### Example: MaxCut

Solving MaxCut on a simple triangle graph (3 nodes).

```python
from metalq.algorithms import QAOA
from metalq import Hamiltonian, Z, run
import networkx as nx

print("=== QAOA: MaxCut on Triangle Graph ===")

# 1. Define Graph & Hamiltonian
# Triangle (0-1, 1-2, 2-0)
edges = [(0, 1), (1, 2), (2, 0)]

# Cost Hamiltonian for MaxCut: H_C = 0.5 * sum_{i,j} (1 - Zi Zj)
# Effectively minimize sum (Zi Zj) to maximize cuts
H_cost = Hamiltonian()
for i, j in edges:
    H_cost = H_cost + (Z(i) * Z(j))

print(f"Graph Edges: {edges}")

# 2. Run QAOA
# p (reps): Number of QAOA layers
print("Running QAOA (p=1)...")
qaoa = QAOA(H_cost, reps=1)
result = qaoa.compute(max_iter=30)

print(f"Optimal Value (Energy): {result.eigenvalue:.4f}")

# 3. Get Solution
# Bind optimal parameters to ansatz and measure
qc_solved = qaoa.ansatz.bind_parameters(result.optimal_params)
final_res = run(qc_solved, shots=1000)
print(f"Measurement Counts: {final_res.counts}")
```

### Output
```text
=== QAOA: MaxCut on Triangle Graph ===
Graph Edges: [(0, 1), (1, 2), (2, 0)]
Running QAOA (p=1)...
Optimal Value (Energy): -0.9999
Measurement Counts: {'010': 184, '110': 171, '101': 151, '000': 1, '011': 158, '100': 163, '001': 172}
```
*Note: The most frequent bitstrings (010, 110, 101, etc.) correspond to cuts that separate the nodes, showing the optimal solution.*
