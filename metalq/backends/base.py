"""
metalq/backends/base.py - Abstract Backend Base Class

すべてのバックエンドが実装すべきインターフェースを定義。
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..circuit import Circuit
    from ..spin import Hamiltonian


class Backend(ABC):
    """
    Abstract base class for quantum simulation backends.
    
    すべてのバックエンド (CPU, MPS) はこのクラスを継承し、
    以下のメソッドを実装する必要がある。
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g., 'cpu', 'mps')."""
        pass
    
    @property
    @abstractmethod
    def max_qubits(self) -> int:
        """Maximum supported qubits (memory-limited)."""
        pass
    
    @abstractmethod
    def run(self, 
            circuit: 'Circuit', 
            shots: int = 0,
            params: Optional[Union[Dict, List[float]]] = None) -> Dict:
        """
        Execute a quantum circuit.
        
        Args:
            circuit: Circuit to execute
            shots: Number of measurement shots.
                   0 = return statevector only
                   >0 = sample and return counts
            params: Parameter values for parameterized circuits
        
        Returns:
            Dict containing:
              - 'statevector': np.ndarray (if shots=0)
              - 'counts': Dict[str, int] (if shots>0)
              - 'time_ms': float (execution time)
        """
        pass
    
    @abstractmethod
    def statevector(self, 
                    circuit: 'Circuit',
                    params: Optional[Union[Dict, List[float]]] = None) -> np.ndarray:
        """
        Get the final statevector after circuit execution.
        
        Args:
            circuit: Circuit to execute (without measurements)
            params: Parameter values
        
        Returns:
            Complex statevector of shape (2^n,)
        """
        pass
    
    @abstractmethod
    def expectation(self, 
                    circuit: 'Circuit', 
                    hamiltonian: 'Hamiltonian',
                    params: Optional[Union[Dict, List[float]]] = None) -> float:
        """
        Compute expectation value <ψ|H|ψ>.
        
        Args:
            circuit: Circuit preparing state |ψ⟩
            hamiltonian: Observable H
            params: Parameter values
        
        Returns:
            Real expectation value
        """
        pass
    
    @abstractmethod
    def gradient(self, 
                 circuit: 'Circuit', 
                 hamiltonian: 'Hamiltonian',
                 params: List[float],
                 method: str = 'parameter_shift') -> np.ndarray:
        """
        Compute gradient of expectation value w.r.t. parameters.
        
        Args:
            circuit: Parameterized circuit
            hamiltonian: Observable
            params: Current parameter values
            method: Differentiation method
                    - 'parameter_shift': 2回路/パラメータ
                    - 'adjoint': GPU向け高速アルゴリズム
        
        Returns:
            Gradient array of shape (num_params,)
        """
        pass
    
    def sample(self, 
               circuit: 'Circuit', 
               shots: int,
               params: Optional[Union[Dict, List[float]]] = None) -> Dict[str, int]:
        """
        Sample measurement results.
        
        Convenience method that calls run() with shots > 0.
        """
        result = self.run(circuit, shots=shots, params=params)
        return result.get('counts', {})
