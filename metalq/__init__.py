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

__version__ = '1.0.0'
