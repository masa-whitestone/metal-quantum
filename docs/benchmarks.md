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

## CPU vs Aer

For smaller systems where GPU overhead dominates, Metal-Q includes an optimized CPU backend with Numba:

| Qubits | Metal-Q CPU | Qiskit Aer | Comparison |
|--------|-------------|------------|------------|
| 10     | 3ms         | 1ms        | Aer faster |
| 14     | 3ms         | 4ms        | **1.31x faster** |
| 16     | 13ms        | 12ms       | Comparable |
