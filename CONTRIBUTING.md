# Contributing to Metal-Q

Thank you for your interest in contributing to Metal-Q!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/masa-whitestone/metal-quantum.git
cd metal-quantum
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

3. Build native libraries:
```bash
make clean && make install
```

## Running Tests

```bash
pytest tests/ -v
```

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
