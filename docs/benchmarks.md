# Performance Benchmarks

Benchmarks on Apple M3 Pro (36GB RAM) demonstrate significant performance improvements over CPU-based simulators. Metal-Q excels particularly with larger qubit counts and deep circuits such as Quantum Fourier Transform (QFT).

!!! note "Note"
    Benchmarks run on Apple M3 Pro (36GB RAM). Metal-Q uses half-precision complex numbers (MPS limit), while Qiskit uses double precision.

## Statevector Simulation

Random circuit simulation performance:

| Qubits | Depth | Metal-Q | Qiskit | Speedup |
|--------|-------|---------|--------|---------|
| 16     | 10    | 2ms     | 43ms   | **17.9x** |
| 20     | 10    | 20ms    | 1025ms | **50.2x** |
| 22     | 10    | 217ms   | 4976ms | **22.9x** |
| 24     | 8     | 775ms   | 16999ms| **21.9x** |
| 26     | 6     | 2510ms  | 54967ms| **21.9x** |

## Quantum Fourier Transform (QFT)

| Qubits | Metal-Q | Qiskit | Speedup |
|--------|---------|--------|---------|
| 16     | 1ms     | 24ms   | **18.6x** |
| 20     | 14ms    | 664ms  | **47.9x** |
| 22     | 137ms   | 3284ms | **23.9x** |
| 24     | 643ms   | 14932ms| **23.2x** |

## Sampling

Measurements with 8192 shots:

| Qubits | Metal-Q | Qiskit Aer | Speedup |
|--------|---------|------------|---------|
| 16     | 9ms     | 16ms       | **1.9x** |
| 20     | 34ms    | 143ms      | **4.2x** |
| 22     | 273ms   | 511ms      | **1.9x** |
| 24     | 974ms   | 1540ms     | **1.6x** |

## CPU Backend

The CPU backend fuses consecutive gates into ≤4-qubit blocks and applies each
block as one Accelerate GEMM (AMX), with a lazily-restored bit-permutation
layout, single-pass diagonal blocks, commutation-aware gate reordering, a
structure-keyed fusion-plan cache, an optional complex64 statevector and
adjoint-method gradients. Gains measured on an Apple M3 Pro, one optimization
step at a time (each row relative to the previous row's code):

| Optimization | Workload | Gain |
|---|---|---|
| Gate fusion + Accelerate GEMM (vs per-gate Numba kernels) | VQE ansatz statevector | 2.3x (12q), 1.9x (16q), 1.6x (18q); QFT 1.2–1.4x |
| complex64 sgemm path, DiagBlock fusion, direct Pauli reduction | VQE objective (energy) | 3.0x (20q), 3.8x (24q) complex64; 2.1x / 2.6x complex128; QAOA 24q up to 4.7x |
| JIT block builder + structure-keyed plan cache | VQE objective, complex64 | 12q 7.2 → 4.3 ms, 16q 11.2 → 8.0 ms |
| Commutation-aware gate reordering | Random circuit statevector, 22q complex128 | 586 → 221 ms; 1.6–2.1x faster than Qiskit Aer at 18–22 qubits |
| Adjoint differentiation as the default CPU gradient | Gradient of a p-parameter circuit | ~3 circuit applications instead of 2p |

Reproduce with `benchmarks/cpu_vs_aer_benchmark.py` (random circuits vs Qiskit
Aer and PennyLane `lightning.qubit`) and `benchmarks/cpu_vqe_benchmark.py`
(statevector / expectation / gradient / fused energy+gradient of a VQE
objective).

