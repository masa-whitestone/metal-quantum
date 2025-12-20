# Metal-Q (metal-quantum)

**Metal-Q** is a high-performance quantum circuit simulator leveraging Apple Silicon's Metal API for GPU acceleration.

> **Project Name**: metal-quantum  
> **Package Name**: metalq

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 2.0+](https://img.shields.io/badge/qiskit-2.0+-6929C4.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **GPU Acceleration**: Uses Metal Compute Shaders for fast statevector simulation
- **Qiskit 2.0+ Integration**: Drop-in replacement for Qiskit backends (simulates `QuantumCircuit` directly)
- **Apple Silicon Native**: Optimized for M1/M2/M3/M4 chips
- **High-Level Gate Support**: QFT, MCX, Custom Unitary, Grover, and more

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- **Qiskit >= 2.0**
- Xcode Command Line Tools

## Installation

```bash
# Prerequisites
xcode-select --install

# Clone and install
git clone https://github.com/masa-whitestone/metal-quantum.git
cd metal-quantum
make install
pip install .
```

## Quick Start

```python
from qiskit import QuantumCircuit
import metalq

# Create a Bell state circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Get statevector (GPU accelerated)
sv = metalq.statevector(qc)
print(sv)  # [0.707+0j, 0+0j, 0+0j, 0.707+0j]

# Run with measurements
qc.measure_all()
result = metalq.run(qc, shots=1024)
print(result.get_counts())  # {'00': ~512, '11': ~512}
```

## Examples

### QFT (Quantum Fourier Transform)
```python
from qiskit.circuit.library import QFTGate

qc = QuantumCircuit(3)
qc.x(0)
qc.append(QFTGate(3), [0, 1, 2])
sv = metalq.statevector(qc)
```

### Multi-Controlled X (MCX)
```python
from qiskit.circuit.library import MCXGate

qc = QuantumCircuit(4)
qc.x([0, 1, 2])  # Set controls
qc.append(MCXGate(3), [0, 1, 2, 3])
sv = metalq.statevector(qc)
```

### Custom Unitary Gate
```python
from qiskit.circuit.library import UnitaryGate
import numpy as np

U = np.array([[np.cos(0.5), -np.sin(0.5)],
              [np.sin(0.5), np.cos(0.5)]])
qc = QuantumCircuit(1)
qc.append(UnitaryGate(U), [0])
sv = metalq.statevector(qc)
```

See [`examples/`](examples/) for more comprehensive examples.

## Supported Gates

| Category | Gates |
|----------|-------|
| 1Q Basic | `id`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, `sx`, `sxdg` |
| 1Q Rotation | `rx`, `ry`, `rz`, `p`, `u`, `u1`, `u2`, `u3`, `r` |
| 2Q | `cx`, `cy`, `cz`, `ch`, `cs`, `csdg`, `csx`, `swap`, `iswap`, `cp`, `crx`, `cry`, `crz`, `cu`, `rxx`, `ryy`, `rzz`, `rzx`, `dcx`, `ecr` |
| 3Q | `ccx`, `ccz`, `cswap` |
| High-Level | `QFTGate`, `MCXGate`, `UnitaryGate`, `DiagonalGate`, `PermutationGate` |

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Python Layer                    │
│              (metalq package)                    │
├─────────────────────────────────────────────────┤
│                 ctypes Bridge                    │
├─────────────────────────────────────────────────┤
│               Native Layer (ObjC)                │
│             libmetalq.dylib                      │
├─────────────────────────────────────────────────┤
│            Metal Compute Shaders                 │
│          quantum_gates.metallib                  │
└─────────────────────────────────────────────────┘
```

## Running Tests

```bash
make install
pip install pytest
pytest tests/
```

## License

MIT