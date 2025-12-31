"""
metalq/backends/cpu/gradient.py - Parameter Shift Gradient

パラメータシフト則を用いた勾配計算。
"""
import numpy as np
from typing import List, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...circuit import Circuit
    from ...spin import Hamiltonian
    from .backend import CPUBackend


def parameter_shift_gradient(backend: 'CPUBackend',
                             circuit: 'Circuit',
                             hamiltonian: 'Hamiltonian',
                             params: List[float],
                             shift: float = np.pi / 2) -> np.ndarray:
    """
    Compute gradient using parameter-shift rule.
    
    各パラメータθについて:
    d<H>/dθ = ( <H>(θ + s) - <H>(θ - s) ) / (2 sin(s))
    
    Args:
        backend: Backend instance to execute circuits
        circuit: Parameterized circuit
        hamiltonian: Observable
        params: Current parameter values
        shift: Shift amount (default: π/2)
        
    Returns:
        Gradient vector
    """
    num_params = len(params)
    grads = np.zeros(num_params)
    
    # Pre-factor for shift=π/2 is 1/2
    factor = 1.0 / (2.0 * np.sin(shift))
    
    # Iterate over each parameter
    # Note: This is sequential and slow. Ideally parallelize.
    for i in range(num_params):
        # Shift forward
        params_plus = params.copy()
        params_plus[i] += shift
        exp_plus = backend.expectation(circuit, hamiltonian, params_plus)
        
        # Shift backward
        params_minus = params.copy()
        params_minus[i] -= shift
        exp_minus = backend.expectation(circuit, hamiltonian, params_minus)
        
        grads[i] = factor * (exp_plus - exp_minus)
        
    return grads
