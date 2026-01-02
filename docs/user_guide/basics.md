# Basics

This guide covers the fundamental usage of Metal-Q for quantum circuit simulation.

## Circuit Construction

The core of Metal-Q is the `Circuit` class. It provides a familiar interface similar to other quantum frameworks.

### Creating a Bell State

```python
from metalq import Circuit

print("=== Basic Circuit Simulation ===")

# Create a Bell State circuit
qc = Circuit(2)
qc.h(0)        # Hadamard gate on qubit 0
qc.cx(0, 1)    # CNOT gate (control 0, target 1)

print("Circuit created:")
print(qc)
```

### Output
```text
=== Basic Circuit Simulation ===
Circuit created:
     ╭─╮   
q_0: │H│─●─
     ╰─╯ │ 
        ╭┴╮
q_1: ───│X│
        ╰─╯
```

## Running Simulation

To simulate the circuit, use the `run` function. Metal-Q automatically selects the best available backend (MPS for Apple Silicon, CPU otherwise), but you can specify it manually.

### MPS Backend (Apple Silicon GPU)

The default backend on macOS utilize Metal Performance Shaders for high-performance GPU simulation.

```python
from metalq import run

# Run on MPS
print("\nRunning on MPS backend...")
result = run(qc, shots=1000, backend='mps')
print(f"Counts: {result.counts}")
print(f"Statevector (first 4 elements): {result.statevector[:4]}")
```

### Output
```text
Running on MPS backend...
Counts: {'11': 501, '00': 499}
Statevector (first 4 elements): [0.70710677+0.j 0.        +0.j 0.        +0.j 0.70710677+0.j]
```

### CPU Backend

Metal-Q also provides a high-performance CPU backend optimized with Numba and Polars.

```python
# Run on CPU
print("\nRunning on CPU backend...")
result_cpu = run(qc, shots=1000, backend='cpu')
print(f"Counts (CPU): {result_cpu.counts}")
```

### Output
```text
Running on CPU backend...
Counts (CPU): {'11': 491, '00': 509}
```

## Qiskit Interoperability

Metal-Q provides seamless adapters to convert Qiskit circuits, allowing you to run existing Qiskit code on Apple Silicon GPUs.

```python
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
```

### Output
```text
=== Qiskit Interoperability ===
Qiskit Circuit created.

Converting to Metal-Q...
Metal-Q Circuit:
     ╭─╮              
q_0: │H│─●────────────
     ╰─╯ │            
        ╭┴╮╭────────╮ 
q_1: ───│X││RZ(0.50)│─
        ╰─╯╰────────╯ 

Running on Metal-Q MPS backend...
Counts: {'00': 54, '11': 46}

Converting back to Qiskit...
Conversion successful.
     ┌───┐                
q_0: ┤ H ├──■─────────────
     └───┘┌─┴─┐┌─────────┐
q_1: ─────┤ X ├┤ Rz(0.5) ├
          └───┘└─────────┘
```
