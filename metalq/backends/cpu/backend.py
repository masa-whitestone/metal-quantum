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
                 max_fused_qubits: int = 4,
                 dtype=np.complex128):
        """Initialize CPU backend.

        Args:
            fusion: Fuse consecutive gates into multi-qubit blocks and
                apply each block as one Accelerate GEMM (AMX-accelerated
                on Apple Silicon). Falls back to per-gate kernels when
                False.
            max_fused_qubits: Maximum qubits per fused block (block
                matrices are 2^k x 2^k).
            dtype: Statevector precision, complex128 (default) or
                complex64. complex64 halves memory traffic and runs the
                fused blocks as sgemm (~2-3.6x faster on Apple Silicon);
                state error stays ~1e-6 for 1000 fused blocks and
                expectation values are always reduced in float64.
                Requires fusion=True.
        """
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
            raise ValueError(
                f"dtype must be complex64 or complex128, got {dtype}")
        if dtype == np.dtype(np.complex64) and not fusion:
            raise ValueError(
                "dtype=complex64 is only supported with fusion=True")
        self._fusion = fusion
        self._max_fused_qubits = max_fused_qubits
        self._dtype = dtype
        # 融合計画のキャッシュ (回路構造キー -> FusionPlan)。VQE の
        # parameter-shift は同一構造を 2p 回評価するため、計画 (DP)
        # を毎回やり直さない。FIFO で上限管理。
        self._plan_cache: Dict = {}
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

        Peak memory = buffers * 2^n * dtype.itemsize bytes. The fused
        path keeps up to 3 statevector-sized buffers alive (input +
        two scratch); the per-gate paths use 2.
        """
        try:
            import psutil
            available_memory = psutil.virtual_memory().available
        except ImportError:
            # Fallback if psutil not installed
            available_memory = 8 * 1024**3  # Assume 8GB

        # Use at most 75% of available memory
        max_memory = available_memory * 0.75

        buffers = 3 if self._fusion else 2
        # Solve: buffers * 2^n * itemsize < max_memory
        import math
        max_n = int(math.log2(max_memory / (buffers * self._dtype.itemsize)))

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
        
        result = {'time_ms': 0.0}

        if shots == 0:
            result['statevector'] = self.statevector(circuit)
        else:
            # Sample from the layout statevector directly; the sampled
            # indices are remapped instead of restoring the full array.
            sv, pos_of = self._statevector_layout(circuit)
            from .measurement import sample_counts
            result['counts'] = sample_counts(
                sv,
                shots,
                circuit.num_qubits,
                use_polars=self._has_polars,
                index_map=pos_of,
            )

        result['time_ms'] = (time.perf_counter() - start_time) * 1000
        return result
    
    def statevector(self,
                    circuit: 'Circuit',
                    params: Optional[Union[Dict, List[float]]] = None) -> np.ndarray:
        """Compute final statevector (canonical qubit order)."""
        # Bind parameters
        if params is not None:
            circuit = circuit.bind_parameters(params)

        # Initialize |0...0⟩
        from .statevector import initialize_statevector, apply_gates

        # Apply gates
        if self._fusion:
            sv = self._apply_fused(circuit, return_layout=False)
        elif self._has_numba:
            from .gates import apply_gates_numba
            sv = initialize_statevector(circuit.num_qubits)
            sv = apply_gates_numba(sv, circuit._gates, circuit.num_qubits)
        else:
            sv = initialize_statevector(circuit.num_qubits)
            sv = apply_gates(sv, circuit._gates, circuit.num_qubits)

        return sv

    def _apply_fused(self, circuit: 'Circuit', return_layout: bool):
        """Run the fused path on |0...0> (parameters already bound)."""
        from .statevector import initialize_statevector
        from .fusion import (make_plan, build_blocks, apply_fused_blocks,
                             _gate_fields)

        key = (circuit.num_qubits, self._max_fused_qubits,
               tuple((name, tuple(qubits))
                     for name, qubits, _ in map(_gate_fields,
                                                circuit._gates)))
        plan = self._plan_cache.get(key)
        if plan is None:
            plan = make_plan(circuit._gates, circuit.num_qubits,
                             self._max_fused_qubits)
            if len(self._plan_cache) >= 32:
                self._plan_cache.pop(next(iter(self._plan_cache)))
            self._plan_cache[key] = plan

        sv = initialize_statevector(circuit.num_qubits, self._dtype)
        blocks = build_blocks(circuit._gates, circuit.num_qubits, plan)
        return apply_fused_blocks(sv, blocks, circuit.num_qubits,
                                  assume_zero=True,
                                  return_layout=return_layout)

    def _statevector_layout(self,
                            circuit: 'Circuit',
                            params: Optional[Union[Dict, List[float]]] = None):
        """Compute the statevector without restoring the qubit layout.

        Returns (sv, pos_of) where pos_of[q] is the index bit currently
        holding qubit q. Skips the final restore permutation; consumers
        (expectation masks, sampling) remap indices instead.
        """
        if params is not None:
            circuit = circuit.bind_parameters(params)

        if not self._fusion:
            sv = self.statevector(circuit)
            return sv, list(range(circuit.num_qubits))

        return self._apply_fused(circuit, return_layout=True)

    def expectation(self,
                    circuit: 'Circuit',
                    hamiltonian: 'Hamiltonian',
                    params: Optional[Union[Dict, List[float]]] = None) -> float:
        """Compute expectation value."""
        from ...spin import PauliTerm, Hamiltonian
        if isinstance(hamiltonian, PauliTerm):
            hamiltonian = Hamiltonian([hamiltonian])

        sv, pos_of = self._statevector_layout(circuit, params)

        # <ψ|H|ψ> term-wise via the direct Pauli index kernel (float64
        # reduction); the permuted layout is absorbed into the masks.
        from .expectation import expectation_layout
        return expectation_layout(sv, hamiltonian, circuit.num_qubits,
                                  pos_of)
    
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
