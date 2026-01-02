"""
Metal-Q High-Level API.
Provides simple functions to run circuits and calculate expectations without managing backend instances manually.
"""
from typing import Optional, List, Dict, Union, Any

from .circuit import Circuit
from .spin import Hamiltonian
from .backends.mps.backend import MPSBackend
from .backends.cpu.backend import CPUBackend

# Global backend instances to avoid recreation overhead
_BACKENDS: Dict[str, Any] = {}

def get_backend(name: str = 'mps'):
    """Get or create a backend instance."""
    if name not in _BACKENDS:
        if name == 'mps':
            try:
                _BACKENDS[name] = MPSBackend()
            except RuntimeError:
                # Fallback if MPS not supported/compiled
                print("Warning: MPS Backend not available, falling back to CPU.")
                _BACKENDS[name] = CPUBackend()
        elif name == 'cpu':
             _BACKENDS[name] = CPUBackend()
        else:
            raise ValueError(f"Unknown backend '{name}'. Available: 'mps', 'cpu'.")
    return _BACKENDS[name]

from .result import Result

# ...

def run(circuit: Circuit, 
        shots: int = 1024, 
        backend: str = 'mps', 
        params: Optional[Union[Dict, List[float]]] = None) -> Result:
    """
    Execute a circuit and return results (counts or statevector).
    
    Args:
        circuit: The quantum circuit to run.
        shots: Number of measurement shots. If 0, returns statevector (if supported).
        backend: Backend name ('mps' or 'cpu').
        params: Parameters to bind to the circuit.
    
    Returns:
        Result object containing 'counts' or 'statevector'.
    """
    bk = get_backend(backend)
    res_dict = bk.run(circuit, shots=shots, params=params)
    
    return Result(
        counts=res_dict.get('counts'),
        statevector=res_dict.get('statevector'),
        metadata={'backend': backend, 'time_ms': res_dict.get('time_ms')}
    )

def expect(circuit: Circuit, 
           hamiltonian: Hamiltonian, 
           backend: str = 'mps', 
           params: Optional[Union[Dict, List[float]]] = None) -> float:
    """
    Calculate the expectation value <psi|H|psi>.
    
    Args:
        circuit: The ansatz circuit.
        hamiltonian: The observable.
        backend: Backend name ('mps' or 'cpu').
        params: Parameters to bind.
        
    Returns:
        Expectation value (float).
    """
    bk = get_backend(backend)
    return bk.expectation(circuit, hamiltonian, params=params)

def statevector(circuit: Circuit, 
                backend: str = 'mps', 
                params: Optional[Union[Dict, List[float]]] = None):
    """
    Get the statevector of a circuit.
    
    Args:
        circuit: The quantum circuit.
        backend: Backend name.
        params: Parameters to bind.
        
    Returns:
        numpy.ndarray: Statevector.
    """
    bk = get_backend(backend)
    return bk.statevector(circuit, params=params)

