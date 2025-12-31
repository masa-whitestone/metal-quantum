"""
examples/vqe_h2.py - VQE for H2 Molecule

This example demonstrates how to use the VQE algorithm in Metal-Q to find the ground state energy
of a Hydrogen molecule (H2) at bond distance 0.74 Angstrom.
"""

import math
from metalq.circuit import Circuit, Parameter
from metalq.spin import Hamiltonian, Z, X
from metalq.algorithms.vqe import VQE
import torch.optim as optim

def main():
    print("=== Metal-Q VQE Example: H2 Molecule ===")

    # 1. Define Hamiltonian for H2 at 0.74A (BK mapping, simplified)
    # H = g0 * I + g1 * Z0 + g2 * Z1 + g3 * Z0Z1 + g4 * X0X1 + g5 * Y0Y1
    # Simplified coefficients (approximate):
    g0 = -0.81261
    g1 = 0.17120
    g2 = -0.22279
    g3 = 0.16862
    g4 = 0.04530
    
    # MetalQ doesn't have Identity term 'I' in Hamiltonian directly usually handled as offset,
    # but we can add constants to the result.
    # We will minimize H_p = g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*X0X1
    # Y0Y1 term is omitted for simplicity in this demo (or mapped to X0X1 via basis rotation).
    
    H_p = (g1 * Z(0)) + (g2 * Z(1)) + (g3 * (Z(0) * Z(1))) + (g4 * (X(0) * X(1)))
    
    print(f"Hamiltonian: {H_p}")
    print(f"Constant shift: {g0}")

    # 2. Define Ansatz (Efficient SU2-like)
    # 2 qubits
    n_qubits = 2
    qc = Circuit(n_qubits)
    
    # Layer 1: Ry Rz
    theta = [Parameter(f'theta_{i}') for i in range(8)]
    
    qc.ry(theta[0], 0)
    qc.rz(theta[1], 0)
    qc.ry(theta[2], 1)
    qc.rz(theta[3], 1)
    
    # Entanglement
    qc.cx(0, 1)
    
    # Layer 2: Ry Rz
    qc.ry(theta[4], 0)
    qc.rz(theta[5], 0)
    qc.ry(theta[6], 1)
    qc.rz(theta[7], 1)
    
    print("\nAnsatz Circuit:")
    print(qc) # Will print representation

    # 3. Setup VQE
    # Use MPS backend to leverage Adjoint Differentiation
    vqe = VQE(ansatz=qc, optimizer_cls=optim.Adam, optimizer_kwargs={'lr': 0.05}, backend='mps')
    
    print("\nStarting Optimization...")
    result = vqe.compute_minimum_eigenvalue(H_p, max_iter=100)
    
    final_energy = result.eigenvalue + g0
    print(f"\nOptimization Complete.")
    print(f"Minimum Energy Found: {final_energy:.6f} Ha")
    print(f"Iterations: {len(result.history)}")
    
    # Reference value for H2 @ 0.74A is roughly -1.137 Ha
    print(f"Reference Energy:   ~ -1.137 Ha (Full CI)")
    
    # Plot history (if interactive)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(result.history, label='VQE Energy (Offset removed)')
        plt.axhline(y=-1.137 - g0, color='r', linestyle='--', label='Reference (shifted)')
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title('VQE Convergence')
        plt.legend()
        plt.savefig('vqe_convergence.png')
        print("Convergence plot saved to 'vqe_convergence.png'")
    except ImportError:
        print("Note: matplotlib not installed, skipping plot generation")

if __name__ == "__main__":
    main()
