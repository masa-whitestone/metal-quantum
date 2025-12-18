# Metal-Q (metal-quantum)

**Metal-Q** is a high-performance quantum circuit simulator leveraging Apple Silicon's Metal API for GPU acceleration.

> **Project Name**: metal-quantum  
> **Package Name**: metalq

## Features
- **GPU Acceleration**: Uses Metal Compute Shaders for fast statevector simulation.
- **Qiskit Integration**: Drop-in replacement for Qiskit backends (simulates `QuantumCircuit` directly).
- **Apple Silicon Native**: Optimized for M1/M2/M3 chips.

## Installation

1. **Prerequisites**
   - macOS with Apple Silicon
   - Xcode Command Line Tools (`xcode-select --install`)
   - Rust/Cargo (for some Python dependencies if building from source, though not strictly required for Metal-Q itself)

2. **Build and Install**
   ```bash
   git clone https://github.com/masa-whitestone/metal-quantum.git
   cd metal-quantum
   make install
   pip install .
   ```

## Usage

```python
from qiskit import QuantumCircuit
import metalq

# Create a circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Run on Metal GPU
result = metalq.run(qc, shots=1024)
print(result.get_counts())
# Output: {'00': 512, '11': 512} (approx)
```

## Architecture
- **Native Layer**: Objective-C + Metal (`libmetalq.dylib`)
- **Python Layer**: `ctypes` wrapper (`metalq` package)

## License
MIT