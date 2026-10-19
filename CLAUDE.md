# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Metal-Q is a high-performance quantum circuit simulator for Apple Silicon (M1-M4). It uses Metal Compute Shaders for GPU-accelerated statevector simulation with native adjoint differentiation, PyTorch integration, and Qiskit compatibility.

## Build & Development

### Native library (required for MPS backend)
```bash
make
```
This delegates to `native/Makefile`, compiles Objective-C and Metal sources under
`native/build/`, and copies `libmetalq.dylib` and `default.metallib` to `metalq/lib/`.
`make install` is an alias for the same build/copy operation. `make clean` removes
both the native build directory and these packaged binaries. See
`docs/installation.md` for Metal toolchain setup.

### Install package (editable)
```bash
python -m pip install -e ".[dev]"
```

### Run tests
```bash
make test                           # build native library, then run all tests
make test PYTHON=python3 PYTEST_ARGS="tests/test_backend_mps.py -q"
python -m pytest tests/test_backend_cpu.py -v  # no native build required
```

### Documentation
```bash
mkdocs serve   # local preview
mkdocs build   # build static site
```

## Architecture

Four-tier stack:

1. **Python API** (`metalq/`) — Circuit builder, parameters, high-level `run`/`expect`/`statevector` API
2. **Backends** (`metalq/backends/`) — Abstract base in `base.py`, two implementations:
   - `cpu/` — Pure NumPy simulator (gates, statevector, measurement, gradient)
   - `mps/` — Metal GPU backend via ctypes → `libmetalq.dylib`
3. **PyTorch integration** (`metalq/torch/`) — `QuantumLayer` (nn.Module) wraps circuits as differentiable layers; `QuantumFunction` implements custom autograd
4. **Native layer** (`native/`) — Objective-C Metal device management + Metal compute shaders for gate kernels

### Key data flow
```
Circuit.bind_parameters() → Backend.run()/statevector()/expectation()
  CPU: NumPy matrix ops
  MPS: ctypes → libmetalq.dylib → Metal GPU shaders
→ Result (counts, statevector, probabilities, expectation)
```

### Important modules
- `circuit.py` — Circuit builder with single/multi-qubit gates, QFT, Grover diffusion
- `parameter.py` — Symbolic parameters supporting arithmetic expressions, used for variational circuits
- `spin.py` — Pauli operator / Hamiltonian construction (`Z(0) @ Z(1) + 0.5*X(0)`)
- `algorithms/` — Ready-to-use VQE and QAOA implementations
- `adapters/qiskit_adapter.py` — Bidirectional Qiskit ↔ Metal-Q circuit conversion

## Platform Constraints

- **macOS 12.0+ with Apple Silicon only** (M1-M4)
- Statevector simulation is memory-exponential; practical limit ~30 qubits
- No noise models (ideal simulation only)
- Requires Xcode Command Line Tools for native build
