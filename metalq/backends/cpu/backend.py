"""
metalq/backends/cpu/backend.py - High-Performance CPU Backend

NumPy + Numba + Polars による高速 CPU シミュレーション。
"""
from typing import Dict, List, Optional, Union, TYPE_CHECKING
import numpy as np
import time

from ..base import Backend

if TYPE_CHECKING:
    from ...circuit import Circuit
    from ...spin import Hamiltonian


class CPUBackend(Backend):
    """
    High-performance CPU backend using NumPy, Numba, and Polars.
    
    最適化戦略:
    - NumPy: ベクトル化された配列演算
    - Numba: JITコンパイルでゲート適用を高速化
    - Polars: 並列集計で測定結果処理を高速化
    """
    
    def __init__(self,
                 fusion: bool = True,
                 max_fused_qubits: int = 4):
        """Initialize CPU backend.

        Args:
            fusion: Fuse consecutive gates into multi-qubit blocks and
                apply each block as one Accelerate GEMM (AMX-accelerated
                on Apple Silicon). Falls back to per-gate kernels when
                False.
            max_fused_qubits: Maximum qubits per fused block (block
                matrices are 2^k x 2^k).
        """
        self._fusion = fusion
        self._max_fused_qubits = max_fused_qubits
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check and import optional dependencies."""
        # Numba (optional but recommended)
        try:
            import numba
            self._has_numba = True
        except ImportError:
            self._has_numba = False
            # import warnings
            # warnings.warn(
            #     "Numba not found. Install with 'pip install numba' for 2-5x speedup."
            # )
        
        # Polars (optional but recommended)
        try:
            import polars
            self._has_polars = True
        except ImportError:
            self._has_polars = False
    
    @property
    def name(self) -> str:
        return 'cpu'
    
    @property
    def max_qubits(self) -> int:
        """
        Maximum qubits based on available memory.
        
        State vector size = 2^n * 16 bytes (complex128)
        """
        try:
            import psutil
            available_memory = psutil.virtual_memory().available
        except ImportError:
            # Fallback if psutil not installed
            available_memory = 8 * 1024**3  # Assume 8GB
        
        # Use at most 75% of available memory
        max_memory = available_memory * 0.75
        
        # Solve: 2^n * 16 < max_memory
        import math
        max_n = int(math.log2(max_memory / 16))
        
        return min(max_n, 30)  # Cap at 30 qubits
    
    # ========================================================================
    # Main Execution Methods
    # ========================================================================
    
    def run(self, 
            circuit: 'Circuit', 
            shots: int = 0,
            params: Optional[Union[Dict, List[float]]] = None) -> Dict:
        """Execute circuit."""
        start_time = time.perf_counter()
        
        # Bind parameters if needed
        if params is not None:
            circuit = circuit.bind_parameters(params)
        
        # Get statevector
        sv = self.statevector(circuit)
        
        result = {'time_ms': 0.0}
        
        if shots == 0:
            result['statevector'] = sv
        else:
            # Sample measurements
            from .measurement import sample_counts
            result['counts'] = sample_counts(
                sv, 
                shots, 
                circuit.num_qubits,
                use_polars=self._has_polars
            )
        
        result['time_ms'] = (time.perf_counter() - start_time) * 1000
        return result
    
    def statevector(self, 
                    circuit: 'Circuit',
                    params: Optional[Union[Dict, List[float]]] = None) -> np.ndarray:
        """Compute final statevector."""
        # Bind parameters
        if params is not None:
            circuit = circuit.bind_parameters(params)
        
        # Initialize |0...0⟩
        from .statevector import initialize_statevector, apply_gates
        sv = initialize_statevector(circuit.num_qubits)
        
        # Apply gates
        if self._fusion:
            from .fusion import apply_gates_fused
            sv = apply_gates_fused(sv, circuit._gates, circuit.num_qubits,
                                   max_fused=self._max_fused_qubits)
        elif self._has_numba:
            from .gates import apply_gates_numba
            sv = apply_gates_numba(sv, circuit._gates, circuit.num_qubits)
        else:
            sv = apply_gates(sv, circuit._gates, circuit.num_qubits)

        return sv
    
    def expectation(self, 
                    circuit: 'Circuit', 
                    hamiltonian: 'Hamiltonian',
                    params: Optional[Union[Dict, List[float]]] = None) -> float:
        """Compute expectation value."""
        from ...spin import PauliTerm, Hamiltonian
        if isinstance(hamiltonian, PauliTerm):
            hamiltonian = Hamiltonian([hamiltonian])

        sv = self.statevector(circuit, params)

        # <ψ|H|ψ> term-wise, without the dense 2^n x 2^n matrix
        from .statevector import expectation_from_statevector
        return expectation_from_statevector(sv, hamiltonian, circuit.num_qubits)
    
    def gradient(self, 
                 circuit: 'Circuit', 
                 hamiltonian: 'Hamiltonian',
                 params: List[float],
                 method: str = 'parameter_shift') -> np.ndarray:
        """Compute gradient using parameter-shift rule."""
        from .gradient import parameter_shift_gradient
        
        return parameter_shift_gradient(
            self, circuit, hamiltonian, params
        )
