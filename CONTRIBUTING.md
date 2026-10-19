# Contributing to Metal-Q

Thank you for your interest in contributing to Metal-Q!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/masa-whitestone/metal-quantum.git
cd metal-quantum
```

2. Activate your Python environment (Conda or venv), then install development
dependencies into that environment. The `dev` extra includes pytest and PyYAML
for the documentation security tests:
```bash
python -m pip install -e ".[dev]"
```

3. Check the [Metal compiler tools](docs/installation.md#metal-compiler-tools),
then build from the repository root:
```bash
make
```

`make` and `make install` both build the native library and shaders under
`native/build/` and copy `libmetalq.dylib` and `default.metallib` to `metalq/lib/`.
Python package installation is handled by the pip command above. To force a
fresh native build, run `make clean` first; this removes the build directory and
both packaged binaries.

## Running Tests

```bash
make test
```

This builds the native libraries before running `python -m pytest tests/ -v`.
Use `PYTHON` to select an interpreter and `PYTEST_ARGS` to select tests or options:

```bash
make test PYTHON=python3 PYTEST_ARGS="tests/test_backend_mps.py -q"
```

To run CPU-only tests without building the native library:

```bash
python -m pytest tests/test_backend_cpu.py -v
```

If you selected a Metal toolchain with `TOOLCHAINS`, use the same environment
variable for `make`, `make install`, and `make test`.

## Code Style

- Python: Follow PEP 8
- Objective-C: Follow Apple's coding guidelines
- Metal: Use descriptive kernel names

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Reporting Issues

Please include:
- macOS version
- Apple Silicon chip (M1/M2/M3/M4)
- Python version
- Minimal reproducible example
