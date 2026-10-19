# Installation

## Requirements

*   **OS**: macOS 12.0+ (Monterey or later)
*   **Hardware**: Apple Silicon (M1/M2/M3/M4) Mac
*   **Python**: 3.10+
*   **Tools**: Xcode with Metal compiler tools (for native compilation)

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
    make
    ```

3.  Install the Python package:
    ```bash
    python -m pip install -e .
    ```

Run these commands from the repository root with your intended Python
environment activated. `make` (or `make install`) builds the native library and
shaders in `native/build/` and copies them to `metalq/lib/`. It does not install
the Python package; use the pip command above for that step.

For contributor dependencies, use `python -m pip install -e ".[dev]"` instead.
See [the contribution guide](https://github.com/masa-whitestone/metal-quantum/blob/main/CONTRIBUTING.md)
for test commands. `make clean` removes the native build directory and the two
packaged binaries, so the next `make` recompiles them.

## Metal Compiler Tools

Check that Xcode is selected and the Metal compiler is available:

```bash
xcode-select -p
xcrun -sdk macosx metal --version
xcrun -sdk macosx --find metallib
```

If Xcode reports a missing Metal Toolchain, install it through Xcode's Components
settings or from Terminal:

```bash
xcodebuild -downloadComponent MetalToolchain
```

If the component is installed but the default compiler lookup still fails,
inspect its status and identifier:

```bash
xcodebuild -showComponent MetalToolchain -json
```

Set `TOOLCHAINS` to the returned `toolchainIdentifier` when building. Replace the
placeholder below with that identifier; its value depends on your Xcode version:

```bash
TOOLCHAINS="identifier-from-toolchainIdentifier" make
```

Use the same variable with `make test` if needed. This selects the toolchain for
that command without changing the system's Xcode selection.

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
