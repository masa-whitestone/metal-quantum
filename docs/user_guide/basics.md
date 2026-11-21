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

Metal-Q provides adapters to convert supported Qiskit circuits for simulation.

`to_metalq` supports the standard gates `id`, `h`, `x`, `y`, `z`, `s`, `sdg`, `t`,
`tdg`, `sx`, `sxdg`, `rx`, `ry`, `rz`, `p`, `cx`, `cz`, `cp`, and `swap`,
plus barriers and measurements. After a measurement, gates can still act on
unmeasured qubits. Standard `delay` instructions are skipped in ideal simulation.
The adapter encodes numeric global phases, including those introduced by
Qiskit's transpiler, with an `rz` and a `p` gate. It omits the pair for phases
within `1e-15` radians of a multiple of 2π.

The adapter preserves global bit indices across registers, the classical bit
count, and each measurement's destination. A circuit imported with zero
classical bits has no destinations for a subsequent Metal-Q `measure_all()`.
Backend sampling and counts formatting are separate from this conversion.

Shared Qiskit parameters, including `ParameterVector` elements, remain shared in
Metal-Q. Bind general expressions such as `2 * theta` in Qiskit using
`assign_parameters` before conversion. Bind symbolic global phases the same way.
Unsupported parameters now raise `metalq.exceptions.ValidationError`. Update
handlers that previously caught `ValueError` to catch this exception. The adapter
also raises it for unsupported operations, classical conditions, and gates that
touch a previously measured qubit. Instruction errors identify the operation,
its zero-based position, and its global qubit indices.

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

### Exporting Metal-Q circuits

`to_qiskit` supports barriers and the following standard gates:

- Single-qubit gates: `id`, `h`, `x`, `y`, `z`, `s`, `sdg`, `t`, `tdg`,
  `sx`, `sxdg`, `rx`, `ry`, `rz`, `p`, `r`, and `u`.
- Two-qubit gates: `cx`, `cy`, `cz`, `ch`, `cp`, `crx`, `cry`, `crz`, `swap`,
  `iswap`, `rxx`, `ryy`, `rzx`, and `rzz`.
- Three-qubit gates: `ccx`, `ccz`, and `cswap`.

Circuits without measurements export with zero classical bits, so they can
be used with Qiskit's `to_gate()` and `control()`. For measured circuits,
the full classical width and all measurement destinations are preserved.
Measurements follow the gates in their original measurement order.

Shared Metal-Q parameters remain shared in Qiskit. After a round trip,
`ParameterVector` elements are ordinary Qiskit parameters. Bind the exported
parameters by name or object because positional binding sorts names such as
`v[10]` before `v[2]`. Distinct parameters with the same name raise
`ValidationError`. Bind them or use unique names before export.
General parameter expressions must first be bound with `Circuit.bind_parameters`.
Unsupported gates also raise `ValidationError`, identifying the operation, its
zero-based gate index, and its qubits. They are never silently omitted. Export
does not modify the input circuit.

```python
from metalq import Circuit, Parameter
from metalq.adapters import to_qiskit

theta = Parameter("theta")
circuit = Circuit(2, 3).rx(theta, 0).rz(theta, 1).measure(1, 2)
exported = to_qiskit(circuit)
assert exported.num_parameters == 1
assert exported.num_clbits == 3
```
