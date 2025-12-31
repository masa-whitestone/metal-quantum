"""
examples/qiskit_interop.py - Qiskit Integration Example

This example demonstrates how to use the Qiskit Adapter to run Qiskit circuits
on the Metal-Q backend.
"""

from metalq.adapters.qiskit_adapter import to_metalq
from metalq.api import run
import numpy as np

def main():
    print("=== Metal-Q Qiskit Integration Example ===")
    
    try:
        from qiskit import QuantumCircuit
        import qiskit

        print(f"Qiskit version: {qiskit.__version__}")
    except ImportError:
        print("Qiskit not installed. Skipping example.")
        return

    # 1. Create a Qiskit Circuit
    # Bell State
    qc_qiskit = QuantumCircuit(2)
    qc_qiskit.h(0)
    qc_qiskit.cx(0, 1)
    qc_qiskit.measure_all()
    
    print("\nOriginal Qiskit Circuit:")
    print(qc_qiskit)
    
    # 2. Convert to Metal-Q Circuit
    print("\nConverting to Metal-Q...")
    mq_circuit = to_metalq(qc_qiskit)
    
    print("\nMetal-Q Circuit:")
    print(mq_circuit)
    
    # 3. Run on Metal-Q (MPS Backend)
    print("\nRunning on Metal-Q (MPS)...")
    res = run(mq_circuit, shots=1000, backend='mps')
    counts = res['counts']
    
    print(f"Counts: {counts}")
    
    # Verify
    # Expect roughly 50% 00 (0) and 50% 11 (3)
    total = sum(counts.values())
    p00 = counts.get('00', 0) / total
    p11 = counts.get('11', 0) / total
    
    print(f"P(00): {p00:.2f}, P(11): {p11:.2f}")
    
    if abs(p00 - 0.5) < 0.1 and abs(p11 - 0.5) < 0.1:
        print("SUCCESS: Bell State created correctly.")
    else:
        print("WARNING: Unexpected distribution.")

if __name__ == "__main__":
    main()

