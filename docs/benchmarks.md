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

### Blocked adjoint sweep and one-pass Hamiltonian reduction

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

Reproduce with `benchmarks/cpu_vs_aer_benchmark.py` (random circuits vs Qiskit
Aer and PennyLane `lightning.qubit`) and `benchmarks/cpu_vqe_benchmark.py`
(statevector / expectation / gradient / fused energy+gradient of a VQE
objective).

