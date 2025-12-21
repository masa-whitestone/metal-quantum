# Metal-Q ⚛️🍎

A high-performance quantum circuit simulator for Apple Silicon, using Metal GPU acceleration.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 2.x](https://img.shields.io/badge/Qiskit-2.x-6929C4.svg)](https://qiskit.org/)

## Overview

Metal-Q is a drop-in replacement for Qiskit's statevector simulator, optimized for Apple Silicon GPUs. It leverages Metal compute shaders to achieve significant speedups on M1/M2/M3/M4 Macs.

### Key Features

- 🚀 **Up to 32x faster** than Qiskit for large circuits
- 🔌 **Drop-in compatible** with Qiskit 2.x circuits
- 🍎 **Native Apple Silicon** optimization via Metal API
- 📦 **Simple installation** - no CUDA or complex dependencies
- ⚛️ **44 quantum gates** supported natively

## Performance

Benchmarked on Apple M3 Pro (16GB RAM):

### Statevector Simulation

| Qubits | Depth | Metal-Q | Qiskit | Speedup |
|--------|-------|---------|--------|---------|
| 16 | 10 | 41ms | 42ms | 1.0x |
| 20 | 10 | 80ms | 1,009ms | **12.7x** |
| 22 | 10 | 265ms | 4,888ms | **18.5x** |
| 24 | 8 | 732ms | 17,285ms | **23.6x** |
| 26 | 6 | 2,303ms | 54,444ms | **23.6x** |

### Sampling (8192 shots)

| Qubits | Metal-Q | Aer | Speedup |
|--------|---------|-----|---------|
| 16 | 15ms | 18ms | 1.2x |
| 20 | 36ms | 138ms | **3.9x** |
| 22 | 203ms | 497ms | **2.5x** |
| 24 | 671ms | 1,475ms | **2.2x** |

### QFT Circuit

| Qubits | Metal-Q | Qiskit | Speedup |
|--------|---------|--------|---------|
| 16 | 20ms | 24ms | 1.2x |
| 20 | 63ms | 603ms | **9.6x** |
| 22 | 137ms | 3,257ms | **23.8x** |
| 24 | 459ms | 14,788ms | **32.2x** |

## Installation

### Requirements

- macOS 12.0+ (Monterey or later)
- Apple Silicon (M1/M2/M3/M4) or Intel Mac with Metal support
- Python 3.9+
- Xcode Command Line Tools

### Install from Source

```bash
git clone https://github.com/masa-whitestone/metal-quantum.git
cd metal-quantum
make install
pip install .
```

## Quick Start

```python
from qiskit import QuantumCircuit
import metalq

# Create a quantum circuit (standard Qiskit)
qc = QuantumCircuit(4, 4)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)
qc.measure([0, 1, 2, 3], [0, 1, 2, 3])

# Run on Metal-Q
result = metalq.run(qc, shots=1024)
print(result.get_counts())
# {'0000': 512, '1111': 512}
```

### Get Statevector

```python
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

# Get statevector directly
statevector = metalq.statevector(qc)
print(statevector)
# [0.707+0j, 0, 0, 0, 0, 0, 0, 0.707+0j]
```

## Supported Gates

### Single-Qubit Gates (19)
`id`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, `sx`, `sxdg`, `rx`, `ry`, `rz`, `p`, `u`, `u1`, `u2`, `u3`, `r`

### Two-Qubit Gates (22)
`cx`, `cy`, `cz`, `ch`, `cs`, `csdg`, `csx`, `cp`, `crx`, `cry`, `crz`, `cu`, `cu1`, `cu3`, `swap`, `iswap`, `dcx`, `ecr`, `rxx`, `ryy`, `rzz`, `rzx`

### Three-Qubit Gates (3)
`ccx` (Toffoli), `cswap` (Fredkin), `ccz`

### High-Level Constructs
QFTGate, MCXGate, GroverOperator, UnitaryGate (automatically decomposed)

## API Reference

### `metalq.run(circuit, shots=1024)`

Execute a quantum circuit with measurements.

```python
result = metalq.run(qc, shots=1024)
counts = result.get_counts()  # {'00': 523, '11': 501}
```

### `metalq.statevector(circuit)`

Get the final statevector without measurements.

```python
sv = metalq.statevector(qc)  # numpy array
```

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
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│              Apple Silicon GPU                   │
│         (M1 / M2 / M3 / M4 Series)              │
└─────────────────────────────────────────────────┘
```

## Limitations

- **macOS only** - Requires Metal API
- **No noise models** - Pure statevector simulation
- **Memory bound** - 28 qubits requires ~2GB GPU memory

## Running Tests

```bash
make install
pip install pytest qiskit-aer
pytest tests/ -v
```

## License

MIT License - see [LICENSE](LICENSE) for details.

---

Made with ❤️ for the quantum computing community on Apple Silicon.
