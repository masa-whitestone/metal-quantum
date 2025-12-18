"""
Metal-Q: Apple Metal GPU Quantum Circuit Simulator
"""

from .backend import get_backend, MetalQBackend
from .result import MetalQResult

__version__ = "0.1.0"
__all__ = ['run', 'statevector', 'MetalQResult', 'MetalQBackend']


def run(circuit, shots: int = 1024) -> MetalQResult:
    """
    Run a quantum circuit on the Metal backend.
    
    Args:
        circuit: Qiskit QuantumCircuit
        shots: Number of shots (default: 1024)
        
    Returns:
        MetalQResult object
    """
    return get_backend().run(circuit, shots)


def statevector(circuit):
    """
    Calculate the statevector of a quantum circuit.
    
    Args:
        circuit: Qiskit QuantumCircuit
        
    Returns:
        numpy.ndarray (complex64)
    """
    return get_backend().statevector(circuit)
