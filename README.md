# Metal-Q

A high-performance quantum circuit optimization and simulation library for Apple Silicon, leveraging Metal GPU acceleration.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Platform macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://developer.apple.com/metal/)

## Overview

Metal-Q is a comprehensive quantum computing library designed specifically for Apple Silicon (M1/M2/M3/M4) devices. Unlike standard simulators, Metal-Q includes a fully differentiable backend (supporting Adjoint Differentiation on GPU) and seamless integration with PyTorch, making it ideal for Quantum Machine Learning (QML) and Variational Algorithms (VQE/QAOA).

### Key Features

*   **GPU Acceleration**: Up to 50x faster than standard CPU simulators for statevector simulation using Metal Compute Shaders.
*   **Adjoint Differentiation**: Native GPU implementation of Adjoint Differentiation, enabling gradient calculation with O(1) memory cost relative to circuit depth, crucial for training large variational circuits.
*   **PyTorch Integration**: Built-in autograd functions allow Metal-Q circuits to act as standard PyTorch layers, enabling hybrid quantum-classical model training.
*   **Algorithms**: Ready-to-use implementations of VQE (Variational Quantum Eigensolver) and QAOA (Quantum Approximate Optimization Algorithm).
*   **Qiskit Compatibility**: Includes a bidirectional adapter to convert circuits to/from Qiskit `QuantumCircuit`.
*   **Native API**: A lightweight, intuitive Python API for circuit construction and execution.

## Installation

### Requirements

*   macOS 12.0+ (Monterey or later)
*   Apple Silicon (M1/M2/M3/M4) Mac
*   Python 3.9+
*   Xcode Command Line Tools

### Install from PyPI

```bash
pip install metalq
```

### Install from Source

```bash
git clone https://github.com/masa-whitestone/metal-quantum.git
cd metal-quantum

# Compile native Metal library
cd native && make && cd ..

# Install Python package
pip install -e .
```

## Quick Start

### 1. Basic Circuit Simulation

Running a simple Bell State circuit using Metal-Q's native API:

```python
from metalq import Circuit, run

# Create a circuit with 2 qubits
qc = Circuit(2)
qc.h(0)
qc.cx(0, 1)

# Run on MPS (Metal Performance Shaders) backend
result = run(qc, shots=1000, backend='mps')

print(f"Counts: {result.counts}")
# Counts: {'00': 502, '11': 498}
```

### 2. Variational Quantum Eigensolver (VQE) with PyTorch

Metal-Q integrates with PyTorch to optimize variational circuits efficiently.

```python
import torch
import torch.optim as optim
from metalq import Circuit, Parameter, Hamiltonian, Z, X
from metalq.torch import QuantumLayer

# Define Hamiltonian: H = Z0 * Z1
H = Z(0) * Z(1)

# Define Ansatz
circuit = Circuit(2)
theta = Parameter('theta')
circuit.rx(theta, 0)
circuit.cx(0, 1)

# Create PyTorch Layer
model = QuantumLayer(circuit, H, backend_name='mps')
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Optimization Loop
for step in range(100):
    optimizer.zero_grad()
    loss = model() # Expectation value
    loss.backward() # Computes gradients via GPU Adjoint Differentiation
    optimizer.step()
    
    if step % 20 == 0:
        print(f"Step {step}, Energy: {loss.item():.4f}")
```

### 3. Qiskit Interoperability

You can create circuits in Qiskit and simulate them on Metal-Q's high-performance backend.

```python
from qiskit import QuantumCircuit
from metalq.adapters.qiskit_adapter import to_metalq, to_qiskit
from metalq import run

# Qiskit Circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Convert to Metal-Q
mq_circuit = to_metalq(qc)

# Run on GPU
result = run(mq_circuit, shots=1000)
print(result.counts)

# Convert back to Qiskit (if needed)
qc_back = to_qiskit(mq_circuit)
```

## Performance

Benchmarks on Apple M3 Pro (36GB RAM) demonstrate significant performance improvements over CPU-based simulators. Metal-Q excels particularly with larger qubit counts and deep circuits such as Quantum Fourier Transform (QFT).

### Statevector Simulation (Random Circuit)

| Qubits | Depth | Metal-Q | Qiskit | Speedup |
|--------|-------|---------|--------|---------|
| 16     | 10    | 2ms     | 43ms   | **17.9x** |
| 20     | 10    | 20ms    | 1025ms | **50.2x** |
| 22     | 10    | 217ms   | 4976ms | **22.9x** |
| 24     | 8     | 775ms   | 16999ms| **21.9x** |
| 26     | 6     | 2510ms  | 54967ms| **21.9x** |

### Quantum Fourier Transform (QFT)

| Qubits | Metal-Q | Qiskit | Speedup |
|--------|---------|--------|---------|
| 16     | 1ms     | 24ms   | **18.6x** |
| 20     | 14ms    | 664ms  | **47.9x** |
| 22     | 137ms   | 3284ms | **23.9x** |
| 24     | 643ms   | 14932ms| **23.2x** |

### Sampling (Shots=8192)

| Qubits | Metal-Q | Qiskit Aer | Speedup |
|--------|---------|------------|---------|
| 16     | 9ms     | 16ms       | **1.9x** |
| 20     | 34ms    | 143ms      | **4.2x** |
| 22     | 273ms   | 511ms      | **1.9x** |
| 24     | 974ms   | 1540ms     | **1.6x** |

*Benchmarks run on Apple M3 Pro (36GB RAM). Metal-Q uses half-precision complex numbers (MPS limit), while Qiskit uses double precision.*

## Documentation

*   **`metalq.Circuit`**: Core class for circuit construction.
*   **`metalq.run(circuit, backend='mps')`**: Execute circuits.
*   **`metalq.expect(circuit, hamiltonian)`**: Calculate expectation values.
*   **`metalq.statevector(circuit)`**: Get the final statevector.
*   **`metalq.torch`**: PyTorch integration modules (`QuantumLayer`, `QuantumFunction`).
*   **`metalq.algorithms`**: VQE and QAOA implementations.

## Examples

Full example scripts are available in the [`examples/`](examples/) directory:

*   **`vqe_h2.py`**: Variational Quantum Eigensolver for H₂ molecule ground state energy
*   **`qaoa_maxcut.py`**: QAOA for solving MaxCut graph optimization
*   **`torch_qnn_classifier.py`**: Quantum Neural Network classifier with PyTorch
*   **`qiskit_interop.py`**: Qiskit interoperability demonstration

## Architecture

Metal-Q is built with a layered architecture to maximize performance while maintaining ease of use:

1.  **Python API**: High-level interface and PyTorch bindings.
2.  **C Interface**: Lightweight Ctypes bridge.
3.  **Objective-C Native Layer**: Manages Metal context and buffers.
4.  **Metal Compute Shaders**: Optimized GPU kernels for gate application, statevector manipulation, and adjoint gradient calculation.

## Limitations

*   **Apple Silicon Only**: Requires macOS devices with Metal support.
*   **Statevector Simulation**: Memory usage grows exponentially (2^N). 30 qubits is the hard limit on most machines (requires ~16GB RAM for statevector).
*   **Noise Models**: v1.0 supports ideal simulation only.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Designed for the quantum future on Apple Silicon.
