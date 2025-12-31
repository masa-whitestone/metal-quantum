# Installation

## Requirements

*   **OS**: macOS 12.0+ (Monterey or later)
*   **Hardware**: Apple Silicon (M1/M2/M3/M4) Mac
*   **Python**: 3.10+
*   **Tools**: Xcode Command Line Tools (for native compilation)

## Install from PyPI

The easiest way to install Metal-Q is via pip:

```bash
pip install metalq
```

## Install from Source

For development or to get the latest features, you can install from source:

1.  Clone the repository:
    ```bash
    git clone https://github.com/masa-whitestone/metal-quantum.git
    cd metal-quantum
    ```

2.  Compile the native Metal library:
    ```bash
    cd native && make && cd ..
    ```

3.  Install the Python package:
    ```bash
    pip install -e .
    ```

## Verification

To verify the installation, run a simple Python script:

```python
from metalq import Circuit, run

# Create a small circuit
qc = Circuit(2)
qc.h(0)
qc.cx(0, 1)

# Run on MPS backend
result = run(qc, shots=100)
print(result.counts)
```

If you see output like `{'00': 48, '11': 52}`, the installation was successful!
