# Metal-Q ⚛️🍎

**A high-performance quantum circuit optimization and simulation library for Apple Silicon, leveraging Metal GPU acceleration.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://developer.apple.com/metal/)
[![PyPI](https://img.shields.io/pypi/v/metalq)](https://pypi.org/project/metalq/)

Metal-Q is a comprehensive quantum computing library designed specifically for Apple Silicon (M1/M2/M3/M4) devices. Unlike standard simulators, Metal-Q includes a fully differentiable backend (supporting Adjoint Differentiation on GPU) and seamless integration with PyTorch, making it ideal for Quantum Machine Learning (QML) and Variational Algorithms (VQE/QAOA).

## Key Features

*   **GPU Acceleration**: Up to 50x faster than standard CPU simulators for statevector simulation using Metal Compute Shaders.
*   **Adjoint Differentiation**: Native GPU implementation of Adjoint Differentiation, enabling gradient calculation with O(1) memory cost relative to circuit depth.
*   **PyTorch Integration**: Built-in autograd functions allow Metal-Q circuits to act as standard PyTorch layers.
*   **Algorithms**: Ready-to-use implementations of VQE (Variational Quantum Eigensolver) and QAOA.
*   **Qiskit Compatibility**: Includes a bidirectional adapter to convert circuits to/from Qiskit `QuantumCircuit`.
*   **Rich Visualization**: Improved terminal-based circuit visualization with Unicode support.

## Installation

```bash
pip install metalq
```

See [Installation](installation.md) for more details.

## Quick Start

Here is a simple example of creating a Bell State using Metal-Q.

```python
from metalq import Circuit, run

# 1. Create a Circuit
qc = Circuit(2)
qc.h(0)
qc.cx(0, 1)

# 2. Visualize
print(qc)
```

**Output:**
```text
     ╭─╮   
q_0: │H│─●─
     ╰─╯ │ 
        ╭┴╮
q_1: ───│X│
        ╰─╯
```

```python
# 3. Run on Metal GPU (MPS)
result = run(qc, shots=1000)
print(f"Counts: {result.counts}")
```

**Output:**
```text
Counts: {'11': 502, '00': 498}
```

## Next Steps

*   **[User Guide](user_guide/basics.md)**: Learn the basics of circuit construction and execution.
*   **[PyTorch Integration](user_guide/pytorch.md)**: Train hybrid quantum-classical models.
*   **[Algorithms](user_guide/algorithms.md)**: Run VQE and QAOA.
*   **[API Reference](api/circuit.md)**: Detailed API documentation.

## License

MIT License. See [LICENSE](license.md) for details.
