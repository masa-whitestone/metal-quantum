# Metal-Q

A high-performance quantum circuit optimization and simulation library for Apple Silicon, leveraging Metal GPU acceleration.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://developer.apple.com/metal/)

## Overview

Metal-Q is a comprehensive quantum computing library designed specifically for Apple Silicon (M1/M2/M3/M4) devices. Unlike standard simulators, Metal-Q includes a fully differentiable backend (supporting Adjoint Differentiation on GPU) and seamless integration with PyTorch, making it ideal for Quantum Machine Learning (QML) and Variational Algorithms (VQE/QAOA).

### Key Features

*   **GPU Acceleration**: 3–20x faster than Qiskit Aer for statevector simulation, expectation values, and sampling using Metal Compute Shaders (measured on Apple M3 Pro, see [Performance](#performance)).
*   **Fused CPU Backend**: Consecutive gates are fused into ≤4-qubit blocks and applied as Accelerate GEMMs on Apple's matrix coprocessor (AMX), with lazy bit-permutation layouts, single-pass diagonal blocks, commutation-aware gate reordering and an optional complex64 path — 1.6–2.1x faster than Qiskit Aer on random 18–22-qubit circuits, and adjoint differentiation on the CPU as well (see [CPU Backend Performance](#cpu-backend-performance)).
*   **GPU-Resident Expectation Values**: Fused Pauli-string kernels evaluate `<psi|H|psi>` entirely on the GPU — one pass per Hamiltonian term, no statevector readback, ~1e-7 agreement with double-precision references.
*   **Adjoint Differentiation**: Native GPU implementation computing all circuit gradients in O(gates) — 25x faster than parameter-shift at 40 parameters. A fused energy+gradient call runs one forward pass per VQE iteration.
*   **GPU-Resident Sampling**: Measurement sampling without reading the statevector back to the CPU (10^6 shots at 24 qubits in ~0.7 s including circuit execution).
*   **PyTorch Integration**: Built-in autograd functions allow Metal-Q circuits to act as standard PyTorch layers, enabling hybrid quantum-classical model training.
*   **Algorithms**: Ready-to-use implementations of VQE (Variational Quantum Eigensolver) and QAOA (Quantum Approximate Optimization Algorithm).
*   **Qiskit Compatibility**: Includes a bidirectional adapter to convert circuits to/from Qiskit `QuantumCircuit`.
*   **Fail-Closed Validation**: Unsupported gates raise a clear `ValidationError` instead of silently corrupting results; all inputs and native-library calls are validated (see [docs/SECURITY.md](docs/SECURITY.md)).

## Installation

### Requirements

*   macOS 12.0+ (Monterey or later)
*   Apple Silicon (M1/M2/M3/M4) Mac
*   Python 3.10+
*   Xcode Command Line Tools (source builds only — the PyPI wheel ships prebuilt binaries)

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

Benchmarks against Qiskit Aer 0.17 (statevector method) on identical circuits.
Measured 2026-07 on an Apple M3 Pro; minimum of repeated runs after warmup.

### Statevector Simulation (3-layer RY/RZ + CX-chain circuit)

| Qubits | Metal-Q | Qiskit Aer | Speedup |
|--------|---------|------------|---------|
| 16     | 1.9ms   | 39ms       | **20.3x** |
| 20     | 6.4ms   | 98ms       | **15.3x** |
| 22     | 66ms    | 284ms      | **4.3x** |
| 24     | 286ms   | 945ms      | **3.3x** |

### Quantum Fourier Transform (QFT)

| Qubits | Metal-Q | Qiskit Aer | Speedup |
|--------|---------|------------|---------|
| 16     | 2.1ms   | 40ms       | **19.3x** |
| 20     | 11ms    | 97ms       | **8.8x** |
| 22     | 133ms   | 286ms      | **2.2x** |
| 24     | 620ms   | 994ms      | **1.6x** |

### Sampling (Shots=8192)

| Qubits | Metal-Q | Qiskit Aer | Speedup |
|--------|---------|------------|---------|
| 16     | 4.6ms   | 46ms       | **10.1x** |
| 20     | 11ms    | 109ms      | **10.2x** |
| 22     | 69ms    | 302ms      | **4.4x** |
| 24     | 289ms   | 967ms      | **3.3x** |

*Metal-Q simulates in single-precision complex (complex64) with GPU reductions
accumulated in double precision — expectation values agree with double-precision
references to ~1e-7 at 24 qubits. Qiskit Aer uses double precision throughout.*

### CPU Backend Performance

The CPU backend (`backend='cpu'`) is not a plain fallback: it fuses consecutive
gates into blocks of at most 4 qubits and applies each block as one real GEMM
(dgemm/sgemm via Accelerate, i.e. on the AMX matrix coprocessor), moves scattered
qubits to the low index bits with a lazily-restored Numba bit-permutation pass,
applies runs of diagonal gates as a single permutation-free phase pass, reorders
commuting gates to pack fusion windows, caches the parameter-independent fusion
plan per circuit structure, and computes gradients with the adjoint method.
`CPUBackend(dtype=np.complex64)` halves memory traffic and runs the blocks as
sgemm; expectation values are always reduced in float64.

Measured on an Apple M3 Pro, one optimization step at a time (each row is the
gain over the previous row's code, from the commit history):

| Optimization | Workload | Gain |
|---|---|---|
| Gate fusion + Accelerate GEMM (vs per-gate Numba kernels) | VQE ansatz statevector | 2.3x (12q), 1.9x (16q), 1.6x (18q); QFT 1.2–1.4x |
| complex64 sgemm path, DiagBlock fusion, direct Pauli reduction | VQE objective (energy) | 3.0x (20q), 3.8x (24q) complex64; 2.1x / 2.6x complex128; QAOA 24q up to 4.7x |
| JIT block builder + structure-keyed plan cache | VQE objective, complex64 | 12q 7.2 → 4.3 ms, 16q 11.2 → 8.0 ms |
| Commutation-aware gate reordering (fused blocks 98 → 39 at 22q) | Random circuit statevector, 22q complex128 | 586 → 221 ms; 1.6–2.1x faster than Qiskit Aer at 18–22 qubits |
| Adjoint differentiation as the default CPU gradient | Gradient of a p-parameter circuit | ~3 circuit applications instead of 2p |

#### Blocked adjoint sweep and one-pass Hamiltonian reduction

The latest round targets the two calls a VQE/QAOA optimizer spends most of its
time in. `expectation` now reduces **all** Hamiltonian terms in one statevector
pass (terms grouped by X/Y mask; Z-only terms need no gather). The adjoint
gradient evaluates each derivative at ψ_j via the small matrix `dU·U†`,
batches the reversed gate list into ≤3-qubit blocks (commuting gates repacked
with the same dependency-preserving reordering used by fusion) and runs
**one Numba kernel per block** that contracts the derivatives and unwinds ψ
and λ together, directly on fusion's permuted layout; diagonal/permutation
gates take a one-multiply path. λ = H_eff ψ and the energy come from a single
write pass, so `CPUBackend.expectation_and_gradient` runs one forward pass
instead of two. At 20 qubits (177 gates, 120 parameters, 39 terms) a gradient
now launches 30 kernels and 118 statevector-sized passes instead of 512 and
1063.

The same change unifies the gradient contract across backends (one entry per
`circuit.parameters`), which fixes `QuantumLayer` on the CPU backend for
circuits that reuse parameters or use expressions such as `rz(2*gamma)`
(previously an `IndexError`), and fixes the parameter-shift fallback for such
expressions (previously returned 0).

Measured on this branch's CI-like environment — a 4-core x86-64 Linux
container with OpenBLAS (`OPENBLAS_NUM_THREADS=1`), no Accelerate/AMX — with
`benchmarks/cpu_vqe_benchmark.py`'s workload (3-layer RY/RZ + CX-chain ansatz,
2n−1-term Ising Hamiltonian), min of 6 runs, before → after:

| dtype | Qubits | statevector | expectation | gradient (adjoint) | expectation_and_gradient |
|---|---|---|---|---|---|
| complex128 | 12 | 1.1 → 1.0 ms | 1.4 → 1.2 ms (**1.17x**) | 4.1 → 3.7 ms (**1.11x**) | 5.3 → 3.7 ms (**1.43x**) |
| complex128 | 16 | 6.3 → 6.0 ms | 7.9 → 7.1 ms (**1.11x**) | 26.4 → 23.0 ms (**1.15x**) | 39.5 → 23.3 ms (**1.70x**) |
| complex128 | 18 | 22.9 → 23.0 ms | 32.2 → 33.3 ms (**0.97x**) | 115.8 → 91.4 ms (**1.27x**) | 160.6 → 86.4 ms (**1.86x**) |
| complex128 | 20 | 68.8 → 67.9 ms | 99.3 → 91.2 ms (**1.09x**) | 500.4 → 343.6 ms (**1.46x**) | 595.7 → 324.3 ms (**1.84x**) |
| complex64 | 12 | 0.9 → 0.9 ms | 1.4 → 1.1 ms (**1.27x**) | 3.8 → 3.9 ms (**0.97x**) | 5.1 → 3.3 ms (**1.55x**) |
| complex64 | 16 | 3.5 → 3.3 ms | 5.5 → 4.3 ms (**1.28x**) | 25.4 → 20.7 ms (**1.23x**) | 29.1 → 20.4 ms (**1.43x**) |
| complex64 | 18 | 11.5 → 10.8 ms | 18.9 → 15.7 ms (**1.20x**) | 99.1 → 72.8 ms (**1.36x**) | 116.7 → 72.8 ms (**1.60x**) |
| complex64 | 20 | 38.5 → 40.0 ms | 80.1 → 65.0 ms (**1.23x**) | 421.5 → 320.0 ms (**1.32x**) | 504.2 → 310.9 ms (**1.62x**) |

Apple Silicon numbers for this round are pending; there the reverse sweep is
DRAM-bound rather than instruction-bound, so the 9x reduction in passes is
expected to matter more (see `docs/ROADMAP.md`).


## Documentation

*   **`metalq.Circuit`**: Core class for circuit construction.
*   **`metalq.run(circuit, backend='mps')`**: Execute circuits (GPU-resident sampling when `shots > 0`).
*   **`metalq.expect(circuit, hamiltonian)`**: Calculate expectation values (GPU-resident fused Pauli kernels).
*   **`metalq.statevector(circuit)`**: Get the final statevector.
*   **`Backend.expectation_and_gradient(circuit, hamiltonian, params)`**: Fused energy + gradients in a single forward pass (adjoint differentiation on both backends).
*   **`metalq.backends.cpu.CPUBackend(fusion=True, max_fused_qubits=4, dtype=np.complex128)`**: Fused Accelerate-GEMM CPU simulator; `dtype=np.complex64` for the single-precision path, `fusion=False` for the per-gate Numba kernels.
*   **`metalq.torch`**: PyTorch integration modules (`QuantumLayer`, `QuantumFunction`).
*   **`metalq.algorithms`**: VQE and QAOA implementations.

See [docs/ROADMAP.md](docs/ROADMAP.md) for known gaps and planned improvements,
and [docs/SECURITY.md](docs/SECURITY.md) for the security model.

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
*   **Statevector Simulation**: Memory usage grows exponentially (2^N). 30 qubits is the enforced limit (~8GB for the complex64 statevector).
*   **Single Precision on GPU**: The GPU statevector is complex64 (GPU reductions accumulate in double precision). The CPU backend defaults to complex128 and offers complex64 as an option; workloads needing full double-precision amplitudes should use the CPU backend.
*   **Noise Models**: v1.0 supports ideal simulation only.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Designed for the quantum future on Apple Silicon.
