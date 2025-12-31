# Basics

## Circuit Construction

The core of Metal-Q is the `Circuit` class. You can initialize it with the number of qubits.

```python
from metalq import Circuit

# 2-qubit circuit
qc = Circuit(2)
```

## Adding Gates

You can add gates by calling methods on the `Circuit` object.

```python
qc.h(0)        # Hadamard on qubit 0
qc.cx(0, 1)    # CNOT with control 0 and target 1
qc.rx(0.5, 0)  # Rotation X by 0.5 radians on qubit 0
```

## Running the Circuit

To simulate the circuit, use the `run` function.

```python
from metalq import run

result = run(qc)
print(result.counts)
```
