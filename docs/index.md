# Metal-Q ⚛️🍎

**A high-performance quantum circuit optimization and simulation library for Apple Silicon, leveraging Metal GPU acceleration.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Platform macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://developer.apple.com/metal/)
[![PyPI](https://img.shields.io/pypi/v/metalq)](https://pypi.org/project/metalq/)

Metal-Q is a comprehensive quantum computing library designed specifically for Apple Silicon (M1/M2/M3/M4) devices. Unlike standard simulators, Metal-Q includes a fully differentiable backend (supporting Adjoint Differentiation on GPU) and seamless integration with PyTorch, making it ideal for Quantum Machine Learning (QML) and Variational Algorithms (VQE/QAOA).

## Key Features

*   **GPU Acceleration**: Up to 50x faster than standard CPU simulators for statevector simulation using Metal Compute Shaders.
*   **Adjoint Differentiation**: Native GPU implementation of Adjoint Differentiation, enabling gradient calculation with O(1) memory cost relative to circuit depth, crucial for training large variational circuits.
*   **PyTorch Integration**: Built-in autograd functions allow Metal-Q circuits to act as standard PyTorch layers, enabling hybrid quantum-classical model training.
*   **Algorithms**: Ready-to-use implementations of VQE (Variational Quantum Eigensolver) and QAOA (Quantum Approximate Optimization Algorithm).
*   **Qiskit Compatibility**: Includes a bidirectional adapter to convert circuits to/from Qiskit `QuantumCircuit`.
*   **Native API**: A lightweight, intuitive Python API for circuit construction and execution.
