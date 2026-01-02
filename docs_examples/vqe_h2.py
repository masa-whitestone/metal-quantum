from metalq import Circuit, Parameter, Hamiltonian, Z, X
from metalq.algorithms import VQE
import numpy as np

print("=== VQE: H2 Molecule (Simplified) ===")

# H2 Hamiltonian at bond distance 0.74A (sto-3g) - Simplified 2-qubit mapping
# H = -1.05 * I + 0.39 * Z0 + 0.39 * Z1 + 0.18 * Z0Z1 + 0.01 * X0X1
H = Hamiltonian()
# Note: metalq.Hamiltonian uses a list of PauliTerms.
# We construct it term by term.
# Constant offset is handled by post-processing usually, but here we focus on observable parts.
H = 0.39 * Z(0) + 0.39 * Z(1) + 0.18 * (Z(0) * Z(1)) + 0.01 * (X(0) * X(1))

print(f"Hamiltonian terms: {len(H.terms)}")

# Ansatz: Hardware Efficient
qc = Circuit(2)
# Parameters
theta = [Parameter(f't{i}') for i in range(4)]

qc.ry(theta[0], 0)
qc.ry(theta[1], 1)
qc.cx(0, 1)
qc.ry(theta[2], 0)
qc.ry(theta[3], 1)

print("Ansatz constructed.")

# Run VQE
# VQE init takes ansatz and optimizer options
vqe = VQE(qc, optimizer_kwargs={'lr': 0.1})

print("\nRunning VQE...")
# compute_minimum_eigenvalue takes hamiltonian
result = vqe.compute_minimum_eigenvalue(H, max_iter=50)

print(f"Minimum Eigenvalue found: {result.eigenvalue:.6f}")
print(f"Optimal Parameters: {[f'{p:.4f}' for p in result.optimal_params]}")
print(f"Total offset (approx -1.05): {result.eigenvalue - 1.05:.6f} Ha")
