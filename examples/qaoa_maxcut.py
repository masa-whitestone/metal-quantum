"""
examples/qaoa_maxcut.py - QAOA for MaxCut

This example demonstrates how to use the QAOA algorithm in Metal-Q to solve the MaxCut problem.
Problem: Divide a graph into two partitions to maximize the number of edges between them.
Formulation: Minimize H = sum_{(i,j) in E} Z_i Z_j.
"""

import numpy as np
import networkx as nx
import torch.optim as optim
from metalq.algorithms.qaoa import QAOA
from metalq.spin import Z

def main():
    print("=== Metal-Q QAOA Example: MaxCut ===")
    
    # 1. Define Graph (Butterfly/Bowtie graph: 0-1, 1-2, 2-3, 3-0 + 1-3 chord? No let's do simple square)
    # 0 -- 1
    # |    |
    # 3 -- 2
    # Edges: (0,1), (1,2), (2,3), (3,0)
    # Optimal cut: {0,2} and {1,3} -> Cuts 4 edges.
    # Energy (Z_i Z_j): 
    # if i,j same spin: +1. cut=0
    # if i,j diff spin: -1. cut=1
    # Max Cut value = sum 0.5 * (1 - Z_i Z_j).
    # Minimize C = Sum Z_i Z_j.
    # Optimal C = -4. (All pairs differ).
    
    n_nodes = 4
    edges = [(0,1), (1,2), (2,3), (3,0)]
    
    print(f"Graph: {n_nodes} nodes, Edges: {edges}")
    
    # 2. Build Hamiltonian
    # H = sum_{i,j} Z_i Z_j
    H = None
    for i, j in edges:
        term = Z(i) * Z(j)
        if H is None:
            H = term
        else:
            H = H + term
            
    print(f"Cost Hamiltonian: {H}")
    
    # 3. Setup QAOA
    p = 2 # Repetitions
    print(f"QAOA Layers (p): {p}")
    
    qaoa = QAOA(hamiltonian=H, reps=p, backend='mps')
    
    # 4. Optimize
    # Use Adam.
    # We can pass custom optimizer details
    qaoa.optimizer_kwargs = {'lr': 0.1}
    
    print("\nStarting Optimization...")
    result = qaoa.compute(max_iter=100)
    
    print(f"\nOptimization Complete.")
    print(f"Minimum Energy Found: {result.eigenvalue:.6f}")
    
    # Check Result
    # Extract optimal parameters and run simulation to get state
    beta_final = result.optimal_params[0::2] # Wait, qaoa builds params gamma_0, beta_0...
    # Actually Parameter order depends on implementation.
    # Let's inspect weights.
    # But usually we just want the statevector or counts.
    
    # We can use the optimized ansatz to get counts
    from metalq.api import run
    
    # Bind parameters
    # The parameters in ansatz are 'gamma_0', 'beta_0', etc.
    # result.optimal_params is a list of floats.
    # We need to map them.
    # VQE returns list of float corresponding to `layer.parameters()`.
    # Pytorch `parameters()` order is usually deterministic (creation order).
    # Creation order: Loop p: gamma, then beta.
    # So [gamma0, beta0, gamma1, beta1...]
    
    param_dict = {}
    idx = 0
    for layer in range(p):
        param_dict[f'gamma_{layer}'] = result.optimal_params[idx]
        idx += 1
        param_dict[f'beta_{layer}'] = result.optimal_params[idx]
        idx += 1
        
    # We need to pass Dictionary mapping Parameter objects or names.
    # MetalQ `run` takes params dict.
    # keys can be string? Let's check api.py -> bind_parameters.
    # It expects Parameter objects usually.
    # But we can find parameters by name from `qaoa.ansatz.parameters`.
    
    # Map name string to Parameter object
    pmap = {p.name: p for p in qaoa.ansatz.parameters}
    bind_map = {pmap[k]: v for k,v in param_dict.items() if k in pmap}
    
    # Run
    res = run(qaoa.ansatz, shots=1000, params=bind_map, backend='mps')
    counts = res['counts']
    
    # Find most frequent bitstring
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    best_bitstring = sorted_counts[0][0]
    
    print(f"\nMost Frequent State: {best_bitstring} (Counts: {sorted_counts[0][1]})")
    
    # Decode Solution
    # 0 and 1 represent partition A and B
    parts = [[], []]
    for i, bit in enumerate(best_bitstring): # Little endian? 
        # MetalQ/Qiskit usually q0 is rightmost? Or array order?
        # MetalQ MPS: q0 is LSB. 
        # String format from `mps/backend.py` -> `run` -> `result.get_counts()`.
        # Native returns counts array index. 
        # If I convert index to string "00...0", index 0 is 000. 1 is 001?
        # Usually q0 is LSB. So "0001" means q0=1.
        # So string is q(N-1)...q0.
        # Let's verify.
        
        # Actually simplest is just to print partitions.
        pass
        
    print(f"Top 5 States: {sorted_counts[:5]}")
    
    # Verify Energy
    # C = sum Z_i Z_j.
    # For "0101" (q3=0, q2=1, q1=0, q0=1).
    # Z0=-1, Z1=1, Z2=-1, Z3=1.
    # (0,1): -1*1 = -1
    # (1,2): 1*-1 = -1
    # (2,3): -1*1 = -1
    # (3,0): 1*-1 = -1
    # Sum = -4. Optimal!
    
    expected_optimal = -4.0
    if result.eigenvalue < -3.5:
        print("SUCCESS: Found near-optimal solution.")
    else:
        print("WARNING: Solution quality low. Maybe more reps needed?")

if __name__ == "__main__":
    main()
