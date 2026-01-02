from qiskit import QuantumCircuit
from metalq.adapters import to_metalq, to_qiskit
from metalq import run

print("=== Qiskit Interoperability ===")

# 1. Create Qiskit Circuit
qc_qiskit = QuantumCircuit(2)
qc_qiskit.h(0)
qc_qiskit.cx(0, 1)
qc_qiskit.rz(0.5, 1)
print("Qiskit Circuit created.")

# 2. Convert to Metal-Q
print("\nConverting to Metal-Q...")
qc_metalq = to_metalq(qc_qiskit)
print(f"Metal-Q Circuit:\n{qc_metalq}")

# 3. Run on Metal-Q
print("\nRunning on Metal-Q MPS backend...")
result = run(qc_metalq, shots=100)
print(f"Counts: {result.counts}")

# 4. Convert back to Qiskit
print("\nConverting back to Qiskit...")
qc_back = to_qiskit(qc_metalq)
print("Conversion successful.")
print(qc_back)
