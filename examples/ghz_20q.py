#!/usr/bin/env python
# coding: utf-8

# # Metal-Q Example: 20-Qubit GHZ State
# 
# This notebook demonstrates the capability of Metal-Q to simulate a 20-qubit GHZ (Greenberger-Horne-Zeilinger) state using the Apple Metal GPU backend.
# 
# Estimated Memory Usage: $2^{20} \times 8$ bytes (complex64) $\approx 8$ MB (Very small for Metal-Q!). 
# Metal-Q can handle up to ~28-30 qubits depending on available RAM.

# In[ ]:


from qiskit import QuantumCircuit
import metalq
import time


# ## Build the Circuit
# We construct a standard GHZ circuit: $H$ on qubit 0, followed by a chain of $CNOT$s.

# In[ ]:


num_qubits = 20
qc = QuantumCircuit(num_qubits)

# H gate on the first qubit
qc.h(0)

# CNOT chain
for i in range(num_qubits - 1):
    qc.cx(i, i + 1)

# Measure all
qc.measure_all()


# ## Run Simulation
# Execute the circuit on the Metal-Q backend.

# In[ ]:


print(f"Simulating {num_qubits} qubits...")

start_time = time.time()
result = metalq.run(qc, shots=1024)
end_time = time.time()

print(f"Simulation completed in {end_time - start_time:.4f} seconds.")


# ## Analyze Results
# For a GHZ state, we expect ideally 50% `00...0` and 50% `11...1`.

# In[ ]:


counts = result.get_counts()

# Filter and show top results
sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
for bitstring, count in sorted_counts[:5]:
    print(f"{bitstring}: {count}")

# Verify GHZ property
zeros = '0' * num_qubits
ones = '1' * num_qubits

total = sum(counts.values())
p_zeros = counts.get(zeros, 0) / total
p_ones = counts.get(ones, 0) / total

print(f"\nProbability |{zeros}>: {p_zeros:.4f}")
print(f"Probability |{ones}>: {p_ones:.4f}")

