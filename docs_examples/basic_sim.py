from metalq import Circuit, run

print("=== Basic Circuit Simulation ===")

# Create a Bell State circuit
qc = Circuit(2)
qc.h(0)
qc.cx(0, 1)

print("Circuit created:")
print(qc) # Will use Visualization if available or just internal repr

# Run on MPS
print("\nRunning on MPS backend...")
result = run(qc, shots=1000, backend='mps')
print(f"Counts: {result.counts}")
print(f"Statevector (first 4 elements): {result.statevector[:4]}")

# Run on CPU
print("\nRunning on CPU backend...")
result_cpu = run(qc, shots=1000, backend='cpu')
print(f"Counts (CPU): {result_cpu.counts}")
