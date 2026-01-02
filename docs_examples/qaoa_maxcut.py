from metalq.algorithms import QAOA
from metalq import Hamiltonian, Z
import networkx as nx

print("=== QAOA: MaxCut on Triangle Graph ===")

# Define Graph: Triangle (0-1, 1-2, 2-0)
# MaxCut solution is 2 cuts (e.g. 0,1 vs 2 -> cuts 0-2 and 1-2)
edges = [(0, 1), (1, 2), (2, 0)]
num_qubits = 3

# Cost Hamiltonian for MaxCut: H_C = 0.5 * sum_{i,j} (1 - Zi Zj)
# We minimize H_C = 0.5 * sum (-Zi Zj) + const -> maximize cuts
# Effectively minimize sum (Zi Zj)
H_cost = Hamiltonian()
for i, j in edges:
    H_cost = H_cost + (Z(i) * Z(j))

print(f"Graph Edges: {edges}")
print("Running QAOA (p=1)...")

qaoa = QAOA(H_cost, reps=1)
result = qaoa.compute(max_iter=30)

print(f"Optimal Value (Energy): {result.eigenvalue:.4f}")

# Get most probable state (solution)
from metalq import run
# Bind optimal parameters to ansatz
qc_solved = qaoa.ansatz.bind_parameters(result.optimal_params)
final_res = run(qc_solved, shots=1000)
print(f"Measurement Counts: {final_res.counts}")
