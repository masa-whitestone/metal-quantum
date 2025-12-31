"""
metalq/backends/cpu/gates.py - Numba JIT Accelerated Gate Operations

Numba による JIT コンパイルでゲート適用を高速化。
ループ処理を C/LLVM レベルに最適化。
"""
import numpy as np
from typing import List, Dict, Any

try:
    import numba
    from numba import jit, prange, complex128, int64, float64
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:
    
    # ========================================================================
    # Numba JIT compiled gate kernels
    # ========================================================================
    
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def _apply_single_qubit_gate_numba(
        sv_real: np.ndarray,
        sv_imag: np.ndarray,
        gate_real: np.ndarray,
        gate_imag: np.ndarray,
        target: int,
        num_qubits: int
    ):
        """
        Apply single-qubit gate using Numba parallel loops.
        
        直接状態ベクトルを操作し、メモリ割り当てを最小化。
        """
        size = 1 << num_qubits
        step = 1 << target
        
        for i in prange(size >> 1):
            # Calculate indices for paired amplitudes
            idx0 = (i // step) * (step << 1) + (i % step)
            idx1 = idx0 + step
            
            # Load current values
            a0_r, a0_i = sv_real[idx0], sv_imag[idx0]
            a1_r, a1_i = sv_real[idx1], sv_imag[idx1]
            
            # Apply 2x2 gate matrix
            # new[0] = gate[0,0] * old[0] + gate[0,1] * old[1]
            # new[1] = gate[1,0] * old[0] + gate[1,1] * old[1]
            
            g00_r, g00_i = gate_real[0, 0], gate_imag[0, 0]
            g01_r, g01_i = gate_real[0, 1], gate_imag[0, 1]
            g10_r, g10_i = gate_real[1, 0], gate_imag[1, 0]
            g11_r, g11_i = gate_real[1, 1], gate_imag[1, 1]
            
            # Complex multiplication for new[0]
            new0_r = (g00_r * a0_r - g00_i * a0_i) + (g01_r * a1_r - g01_i * a1_i)
            new0_i = (g00_r * a0_i + g00_i * a0_r) + (g01_r * a1_i + g01_i * a1_r)
            
            # Complex multiplication for new[1]
            new1_r = (g10_r * a0_r - g10_i * a0_i) + (g11_r * a1_r - g11_i * a1_i)
            new1_i = (g10_r * a0_i + g10_i * a0_r) + (g11_r * a1_i + g11_i * a1_r)
            
            sv_real[idx0] = new0_r
            sv_imag[idx0] = new0_i
            sv_real[idx1] = new1_r
            sv_imag[idx1] = new1_i
    
    
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def _apply_two_qubit_gate_numba(
        sv_real: np.ndarray,
        sv_imag: np.ndarray,
        gate_real: np.ndarray,
        gate_imag: np.ndarray,
        control: int,
        target: int,
        num_qubits: int
    ):
        """Apply two-qubit gate using Numba."""
        size = 1 << num_qubits
        
        # Ensure control < target for index calculation
        if control > target:
            # We need to swap control/target logic or matrix. 
            # For general 2-qubit gate, simple index swapping is tricky without matrix swap.
            # But here we assume `gate_real/imag` is already for `control, target` order logic.
            # Let's standardise index calculation.
            pass
        
        # Consistent step calculation
        step_c = 1 << control
        step_t = 1 << target
        
        # We iterate over size/4 blocks.
        # This loop logic needs to be careful to hit all unique quartets.
        
        # Better approach for generic 2Q iteration:
        # Iterate i from 0 to size/4
        # Insert 0 bits at control and target positions to get base index
        
        for i in prange(size >> 2):
            base = i
            
            # Insert zero at lower position first
            pos1 = min(control, target)
            pos2 = max(control, target)
            step1 = 1 << pos1
            step2 = 1 << pos2
            
            # Insert at pos1
            base = (base // step1) * (step1 << 1) + (base % step1)
            # Insert at pos2
            base = (base // step2) * (step2 << 1) + (base % step2)
            
            # Four amplitudes in standard lexical order 00, 01, 10, 11
            # But we need verify which bit is control/target for indexing purposes?
            # Actually, standard order is:
            # if control < target:
            #   00: base
            #   01: base + step_t
            #   10: base + step_c
            #   11: base + step_c + step_t
            # Wait, 01 means control=0 target=1? 
            # Usual convention: qubit 0 is LSB or MSB?
            # In Qiskit/MetalQ: q0 is LSB (usual python bitwise).
            
            # Let's conform to standard basis: |q_high ... q_low>
            # But the gate matrix is usually given as |00> |01> |10> |11> (tensor product order)
            # which usually means q_control \otimes q_target if that's the gate definintion.
            
            idx00 = base
            idx01 = base + step_t
            idx10 = base + step_c
            idx11 = base + step_c + step_t
            
            # Load values
            a00_r, a00_i = sv_real[idx00], sv_imag[idx00]
            # We need to be careful about matrix index vs qubit index
            # If standard matrix is defined as |c t>, then:
            # 00 -> c=0, t=0
            # 01 -> c=0, t=1
            # 10 -> c=1, t=0
            # 11 -> c=1, t=1
            
            # So the indices above are correct for |ct> order IF control is higher significance?
            # Actually no, bitwise indices:
            # step_c corresponds to bit `control`
            # step_t corresponds to bit `target`
            
            # If we assume gate matrix is 4x4 with rows/cols 0..3 corresponding to 00, 01, 10, 11 of (control, target)
            # then we must load them in that order.
            
            # Case 1: control > target (control is more significant)
            # Then |c t> corresponds to integer c*2 + t
            # 0 -> 00 -> base
            # 1 -> 01 -> base + step_t
            # 2 -> 10 -> base + step_c
            # 3 -> 11 -> base + step_c + step_t
            
            # Case 2: target > control (target is more significant)
            # Then |c t> corresponds to c*2 + t ?? No, usually generic gate is defined on wires q1, q2
            # and matrix matches q1 \otimes q2.
            # So if we apply GenericGate(q_c, q_t), we expect matrix rows to map to state |q_c q_t>.
            
            # Correct mapping:
            # We want to map vector [v00, v01, v10, v11] where indices are determined by state of (q_c, q_t).
            # v00: q_c=0, q_t=0 -> base
            # v01: q_c=0, q_t=1 -> base + step_t
            # v10: q_c=1, q_t=0 -> base + step_c
            # v11: q_c=1, q_t=1 -> base + step_c + step_t
            
            val00_r, val00_i = sv_real[idx00], sv_imag[idx00]
            val01_r, val01_i = sv_real[idx01], sv_imag[idx01]
            val10_r, val10_i = sv_real[idx10], sv_imag[idx10]
            val11_r, val11_i = sv_real[idx11], sv_imag[idx11]
            
            # Apply 4x4
            # We do it manually unrolled for 4x4 to avoid small loops overhead
            # Row 0
            res00_r = gate_real[0,0]*val00_r - gate_imag[0,0]*val00_i + gate_real[0,1]*val01_r - gate_imag[0,1]*val01_i + gate_real[0,2]*val10_r - gate_imag[0,2]*val10_i + gate_real[0,3]*val11_r - gate_imag[0,3]*val11_i
            res00_i = gate_real[0,0]*val00_i + gate_imag[0,0]*val00_r + gate_real[0,1]*val01_i + gate_imag[0,1]*val01_r + gate_real[0,2]*val10_i + gate_imag[0,2]*val10_r + gate_real[0,3]*val11_i + gate_imag[0,3]*val11_r

            # Row 1
            res01_r = gate_real[1,0]*val00_r - gate_imag[1,0]*val00_i + gate_real[1,1]*val01_r - gate_imag[1,1]*val01_i + gate_real[1,2]*val10_r - gate_imag[1,2]*val10_i + gate_real[1,3]*val11_r - gate_imag[1,3]*val11_i
            res01_i = gate_real[1,0]*val00_i + gate_imag[1,0]*val00_r + gate_real[1,1]*val01_i + gate_imag[1,1]*val01_r + gate_real[1,2]*val10_i + gate_imag[1,2]*val10_r + gate_real[1,3]*val11_i + gate_imag[1,3]*val11_r

            # Row 2
            res10_r = gate_real[2,0]*val00_r - gate_imag[2,0]*val00_i + gate_real[2,1]*val01_r - gate_imag[2,1]*val01_i + gate_real[2,2]*val10_r - gate_imag[2,2]*val10_i + gate_real[2,3]*val11_r - gate_imag[2,3]*val11_i
            res10_i = gate_real[2,0]*val00_i + gate_imag[2,0]*val00_r + gate_real[2,1]*val01_i + gate_imag[2,1]*val01_r + gate_real[2,2]*val10_i + gate_imag[2,2]*val10_r + gate_real[2,3]*val11_i + gate_imag[2,3]*val11_r

            # Row 3
            res11_r = gate_real[3,0]*val00_r - gate_imag[3,0]*val00_i + gate_real[3,1]*val01_r - gate_imag[3,1]*val01_i + gate_real[3,2]*val10_r - gate_imag[3,2]*val10_i + gate_real[3,3]*val11_r - gate_imag[3,3]*val11_i
            res11_i = gate_real[3,0]*val00_i + gate_imag[3,0]*val00_r + gate_real[3,1]*val01_i + gate_imag[3,1]*val01_r + gate_real[3,2]*val10_i + gate_imag[3,2]*val10_r + gate_real[3,3]*val11_i + gate_imag[3,3]*val11_r
            
            sv_real[idx00] = res00_r
            sv_imag[idx00] = res00_i
            sv_real[idx01] = res01_r
            sv_imag[idx01] = res01_i
            sv_real[idx10] = res10_r
            sv_imag[idx10] = res10_i
            sv_real[idx11] = res11_r
            sv_imag[idx11] = res11_i


    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def _apply_x_gate_numba(sv_real: np.ndarray, sv_imag: np.ndarray, 
                            target: int, num_qubits: int):
        """Optimized X gate (just swap pairs)."""
        size = 1 << num_qubits
        step = 1 << target
        
        for i in prange(size >> 1):
            idx0 = (i // step) * (step << 1) + (i % step)
            idx1 = idx0 + step
            
            # Swap
            # sv_real[idx0], sv_real[idx1] = sv_real[idx1], sv_real[idx0] # Multiprocessing safe? 
            # Parallel assignment is safe here because idx0/idx1 pairs are distinct for each i.
            
            tmp_r = sv_real[idx0]
            sv_real[idx0] = sv_real[idx1]
            sv_real[idx1] = tmp_r
            
            tmp_i = sv_imag[idx0]
            sv_imag[idx0] = sv_imag[idx1]
            sv_imag[idx1] = tmp_i
    
    
    @jit(nopython=True, parallel=True, cache=True)
    def _apply_z_gate_numba(sv_real: np.ndarray, sv_imag: np.ndarray,
                            target: int, num_qubits: int):
        """Optimized Z gate (negate |1⟩ amplitudes)."""
        size = 1 << num_qubits
        step = 1 << target
        
        for i in prange(size >> 1):
            idx0 = (i // step) * (step << 1) + (i % step)
            idx1 = idx0 + step
            
            # Negate |1⟩
            sv_real[idx1] = -sv_real[idx1]
            sv_imag[idx1] = -sv_imag[idx1]
    
    
    @jit(nopython=True, parallel=True, cache=True)
    def _apply_h_gate_numba(sv_real: np.ndarray, sv_imag: np.ndarray,
                            target: int, num_qubits: int):
        """Optimized Hadamard gate."""
        size = 1 << num_qubits
        step = 1 << target
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        
        for i in prange(size >> 1):
            idx0 = (i // step) * (step << 1) + (i % step)
            idx1 = idx0 + step
            
            a0_r, a0_i = sv_real[idx0], sv_imag[idx0]
            a1_r, a1_i = sv_real[idx1], sv_imag[idx1]
            
            # H = (1/√2) [[1, 1], [1, -1]]
            sv_real[idx0] = inv_sqrt2 * (a0_r + a1_r)
            sv_imag[idx0] = inv_sqrt2 * (a0_i + a1_i)
            sv_real[idx1] = inv_sqrt2 * (a0_r - a1_r)
            sv_imag[idx1] = inv_sqrt2 * (a0_i - a1_i)
    
    
    @jit(nopython=True, parallel=True, cache=True)
    def _apply_cx_gate_numba(sv_real: np.ndarray, sv_imag: np.ndarray,
                             control: int, target: int, num_qubits: int):
        """Optimized CNOT gate (conditional swap)."""
        size = 1 << num_qubits
        step_c = 1 << control
        step_t = 1 << target
        
        for i in prange(size >> 2):
            # Skip if control bit is 0
            # Only operate on |1x⟩ → |1(x⊕1)⟩
            
            base = i
            # Insert 0 at control position
            # Logic to skip to where control=1?
            # Actually easier: iterate base as usual (size >> 2) which covers 00, 01, 10, 11 of other qubits
            # Then reconstruct full index by inserting 1 at control, and 0/1 at target
            
            # Better logic to match _apply_two_qubit general logic:
            # We iterate i from 0 to size/4.
            # Convert i to index with 0 at control and 0 at target.
            
            pos1 = min(control, target)
            pos2 = max(control, target)
            step1 = 1 << pos1
            step2 = 1 << pos2
            
            base = (base // step1) * (step1 << 1) + (base % step1)
            base = (base // step2) * (step2 << 1) + (base % step2)
            
            # Now we have basis where control=0, target=0
            # But we only want to act when control=1
            # The four indices are:
            # 00: c=0, t=0
            # 01: c=0, t=1 (if t<c) or c=0, t=1 (if c<t) -> wait, step_t is associated with target bit
            
            # We want indices where control bit is 1.
            # Corresponding offsets are step_c and step_c + step_t.
            
            idx10 = base + step_c           # Control=1, Target=0
            idx11 = base + step_c + step_t  # Control=1, Target=1
            
            # Swap target bit when control is 1
            tmp_r = sv_real[idx10]
            sv_real[idx10] = sv_real[idx11]
            sv_real[idx11] = tmp_r
            
            tmp_i = sv_imag[idx10]
            sv_imag[idx10] = sv_imag[idx11]
            sv_imag[idx11] = tmp_i


    def apply_gates_numba(sv: np.ndarray, gates: List, num_qubits: int) -> np.ndarray:
        """
        Apply gates using Numba-accelerated kernels.
        
        Args:
            sv: Complex statevector
            gates: List of Gate objects or dicts
            num_qubits: Number of qubits
        
        Returns:
            Updated statevector
        """
        # Split into real/imag for Numba (避免 complex 类型问题)
        sv_real = sv.real.copy()
        sv_imag = sv.imag.copy()
        
        from .statevector import get_gate_matrix
        
        for gate in gates:
            name = gate.name if hasattr(gate, 'name') else gate['name']
            qubits = gate.qubits if hasattr(gate, 'qubits') else gate['qubits']
            params = gate.params if hasattr(gate, 'params') else gate.get('params', [])
            
            if name == 'barrier':
                continue
            
            # Use optimized kernels for common gates
            if name == 'x' and len(qubits) == 1:
                _apply_x_gate_numba(sv_real, sv_imag, qubits[0], num_qubits)
            elif name == 'z' and len(qubits) == 1:
                _apply_z_gate_numba(sv_real, sv_imag, qubits[0], num_qubits)
            elif name == 'h' and len(qubits) == 1:
                _apply_h_gate_numba(sv_real, sv_imag, qubits[0], num_qubits)
            elif name in ('cx', 'cnot') and len(qubits) == 2:
                _apply_cx_gate_numba(sv_real, sv_imag, qubits[0], qubits[1], num_qubits)
            elif len(qubits) == 1:
                # Generic single-qubit gate
                matrix = get_gate_matrix(name, params)
                _apply_single_qubit_gate_numba(
                    sv_real, sv_imag,
                    matrix.real.astype(np.float64),
                    matrix.imag.astype(np.float64),
                    qubits[0], num_qubits
                )
            elif len(qubits) == 2:
                # Generic two-qubit gate
                matrix = get_gate_matrix(name, params)
                _apply_two_qubit_gate_numba(
                    sv_real, sv_imag,
                    matrix.real.astype(np.float64),
                    matrix.imag.astype(np.float64),
                    qubits[0], qubits[1], num_qubits
                )
            else:
                # Fallback to NumPy for 3+ qubit gates
                # Reconstruct complex sv
                sv_tmp = sv_real + 1j * sv_imag
                from .statevector import apply_gate
                sv_tmp = apply_gate(sv_tmp, gate, num_qubits)
                sv_real = sv_tmp.real.copy()
                sv_imag = sv_tmp.imag.copy()
        
        return sv_real + 1j * sv_imag

else:
    # Numba not available - provide fallback
    def apply_gates_numba(sv: np.ndarray, gates: List, num_qubits: int) -> np.ndarray:
        """Fallback to pure NumPy when Numba is not available."""
        from .statevector import apply_gates
        return apply_gates(sv, gates, num_qubits)
