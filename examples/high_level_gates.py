"""
high_level_gates.py - Metal-Q High-Level Gate Examples

Demonstrates:
- QFT (Quantum Fourier Transform)
- Multi-Controlled X (MCX)
- Custom Unitary Gates
- Controlled-Phase Gates (CP/CZ)
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate, MCXGate, UnitaryGate
from qiskit.quantum_info import Statevector
import metalq


def example_qft():
    """3-qubit QFT example"""
    print("=" * 50)
    print("QFT (Quantum Fourier Transform) Example")
    print("=" * 50)
    
    qc = QuantumCircuit(3)
    qc.x(0)  # Start with |001>
    qc.append(QFTGate(3), [0, 1, 2])
    
    sv_metalq = metalq.statevector(qc)
    sv_qiskit = Statevector(qc).data
    
    print(f"Input: |001>")
    print(f"Metal-Q result (magnitude): {np.abs(sv_metalq)}")
    print(f"Qiskit result (magnitude):  {np.abs(sv_qiskit)}")
    print(f"Match: {np.allclose(np.abs(sv_metalq), np.abs(sv_qiskit), atol=1e-4)}")
    print()


def example_mcx():
    """Multi-Controlled X (Toffoli generalization) example"""
    print("=" * 50)
    print("MCX (Multi-Controlled X) Example")
    print("=" * 50)
    
    # 4-qubit circuit: 3 controls + 1 target
    qc = QuantumCircuit(4)
    qc.x([0, 1, 2])  # Set all controls to 1
    qc.append(MCXGate(3), [0, 1, 2, 3])  # Flip target if all controls are 1
    
    sv_metalq = metalq.statevector(qc)
    sv_qiskit = Statevector(qc).data
    
    print(f"Controls: q0=1, q1=1, q2=1 -> Target q3 should flip")
    print(f"Metal-Q result: {np.abs(sv_metalq)}")
    print(f"Expected: |1111> = index 15")
    print(f"Match: {np.allclose(np.abs(sv_metalq), np.abs(sv_qiskit), atol=1e-3)}")
    print()


def example_custom_unitary():
    """Custom Unitary Gate example"""
    print("=" * 50)
    print("Custom Unitary Gate Example")
    print("=" * 50)
    
    # Random 2x2 unitary (Hadamard-like rotation)
    theta = np.pi / 6
    U = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ], dtype=complex)
    
    qc = QuantumCircuit(1)
    qc.append(UnitaryGate(U), [0])
    
    sv_metalq = metalq.statevector(qc)
    sv_qiskit = Statevector(qc).data
    
    print(f"Custom Unitary: Rotation by {np.degrees(theta):.1f} degrees")
    print(f"Metal-Q result: {sv_metalq}")
    print(f"Qiskit result:  {sv_qiskit}")
    print(f"Match: {np.allclose(np.abs(sv_metalq), np.abs(sv_qiskit), atol=1e-4)}")
    print()


def example_controlled_phase():
    """Controlled-Phase (CZ) Gate example"""
    print("=" * 50)
    print("Controlled-Phase (CP/CZ) Gate Example")
    print("=" * 50)
    
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)  # Create |++> superposition
    qc.cp(np.pi, 0, 1)  # CZ: flip sign of |11>
    
    sv_metalq = metalq.statevector(qc)
    expected = np.array([0.5, 0.5, 0.5, -0.5], dtype=complex)
    
    print(f"Initial: |++> = 0.5 * [1, 1, 1, 1]")
    print(f"After CZ: 0.5 * [1, 1, 1, -1]")
    print(f"Metal-Q result: {sv_metalq}")
    print(f"Expected:       {expected}")
    print(f"Match: {np.allclose(sv_metalq, expected, atol=1e-4)}")
    print()


def example_grover_iteration():
    """Single Grover iteration example"""
    print("=" * 50)
    print("Grover's Algorithm (1 iteration) Example")
    print("=" * 50)
    
    n = 3
    qc = QuantumCircuit(n)
    
    # Superposition
    qc.h(range(n))
    
    # Oracle: mark |111>
    qc.x(range(n))
    qc.h(n-1)
    qc.mcx(list(range(n-1)), n-1)
    qc.h(n-1)
    qc.x(range(n))
    
    # Diffuser
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n-1)
    qc.mcx(list(range(n-1)), n-1)
    qc.h(n-1)
    qc.x(range(n))
    qc.h(range(n))
    
    sv_metalq = metalq.statevector(qc)
    sv_qiskit = Statevector(qc).data
    
    probs_metalq = np.abs(sv_metalq) ** 2
    probs_qiskit = np.abs(sv_qiskit) ** 2
    
    print(f"Target state: |000> (marked)")
    print(f"Metal-Q probabilities: {probs_metalq}")
    print(f"Qiskit probabilities:  {probs_qiskit}")
    print(f"|000> probability (Metal-Q): {probs_metalq[0]:.4f}")
    print(f"Match: {np.allclose(probs_metalq, probs_qiskit, atol=1e-3)}")
    print()


if __name__ == "__main__":
    print("\n🚀 Metal-Q High-Level Gates Demo\n")
    
    example_qft()
    example_mcx()
    example_custom_unitary()
    example_controlled_phase()
    example_grover_iteration()
    
    print("✅ All examples completed!")
