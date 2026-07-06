"""
Metal-Q: High-Performance Quantum Circuit Simulator for Apple Silicon
"""
from .parameter import Parameter, ParameterExpression
from .circuit import Circuit, Gate
from .result import Result
from .spin import PauliTerm, Hamiltonian, X, Y, Z, I, Sx, Sy, Sz, Si
from .visualization import draw_circuit
from .api import run, expect, statevector

__all__ = [
    'Parameter',
    'ParameterExpression',
    'Circuit',
    'Gate',
    'Result',
    'PauliTerm',
    'Hamiltonian',
    'X', 'Y', 'Z', 'I',
    'Sx', 'Sy', 'Sz', 'Si',
    'draw_circuit',
    'run',
    'expect',
    'statevector',
]

# Single source of truth is pyproject.toml; read the installed metadata so
# __version__ can never drift from the released version again.
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("metalq")
except Exception:  # uninstalled source tree (e.g. PYTHONPATH usage)
    __version__ = "0.0.0+unknown"
