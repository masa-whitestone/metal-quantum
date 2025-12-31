"""
metalq/backends/cpu/fast_gates.py - Optimized Gate Operations

Highly optimized single/two-qubit gate kernels with:
- Specialized implementations for common gates (X, H, CX, etc.)
- Cache-friendly memory access patterns
- Reduced overhead for small circuits
"""
import numpy as np

try:
    from numba import jit, prange, complex128
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:
    
    # ===========================================================================
    # Specialized Single-Qubit Gates (No matrix multiplication overhead)
    # ===========================================================================
    
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def apply_x_gate(sv: np.ndarray, target: int, num_qubits: int):
        """Optimized Pauli-X (bit flip) - no complex mult needed."""
        step = 1 << target
        size = 1 << num_qubits
        
        for i in prange(size >> 1):
            idx0 = (i // step) * (step << 1) + (i % step)
            idx1 = idx0 + step
            
            # Swap: just exchange values
            sv[idx0], sv[idx1] = sv[idx1], sv[idx0]
    
    
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def apply_z_gate(sv: np.ndarray, target: int, num_qubits: int):
        """Optimized Pauli-Z (phase flip) - just negate |1⟩ component."""
        step = 1 << target
        size = 1 << num_qubits
        
        for i in prange(size >> 1):
            idx1 = (i // step) * (step << 1) + (i % step) + step
            sv[idx1] = -sv[idx1]
    
    
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def apply_h_gate(sv: np.ndarray, target: int, num_qubits: int):
        """Optimized Hadamard gate."""
        step = 1 << target
        size = 1 << num_qubits
        inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
        
        for i in prange(size >> 1):
            idx0 = (i // step) * (step << 1) + (i % step)
            idx1 = idx0 + step
            
            a0, a1 = sv[idx0], sv[idx1]
            
            sv[idx0] = (a0 + a1) * inv_sqrt2
            sv[idx1] = (a0 - a1) * inv_sqrt2
    
    
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def apply_s_gate(sv: np.ndarray, target: int, num_qubits: int):
        """Optimized S gate (phase = i)."""
        step = 1 << target
        size = 1 << num_qubits
        
        for i in prange(size >> 1):
            idx1 = (i // step) * (step << 1) + (i % step) + step
            sv[idx1] = sv[idx1] * 1j
    
    
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def apply_t_gate(sv: np.ndarray, target: int, num_qubits: int):
        """Optimized T gate."""
        step = 1 << target
        size = 1 << num_qubits
        phase = 0.7071067811865476 + 0.7071067811865476j  # exp(i*pi/4)
        
        for i in prange(size >> 1):
            idx1 = (i // step) * (step << 1) + (i % step) + step
            sv[idx1] = sv[idx1] * phase
    
    
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def apply_rx_gate(sv: np.ndarray, theta: float, target: int, num_qubits: int):
        """Optimized RX gate."""
        step = 1 << target
        size = 1 << num_qubits
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        
        for i in prange(size >> 1):
            idx0 = (i // step) * (step << 1) + (i % step)
            idx1 = idx0 + step
            
            a0, a1 = sv[idx0], sv[idx1]
            
            sv[idx0] = c * a0 - 1j * s * a1
            sv[idx1] = -1j * s * a0 + c * a1
    
    
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def apply_ry_gate(sv: np.ndarray, theta: float, target: int, num_qubits: int):
        """Optimized RY gate."""
        step = 1 << target
        size = 1 << num_qubits
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        
        for i in prange(size >> 1):
            idx0 = (i // step) * (step << 1) + (i % step)
            idx1 = idx0 + step
            
            a0, a1 = sv[idx0], sv[idx1]
            
            sv[idx0] = c * a0 - s * a1
            sv[idx1] = s * a0 + c * a1
    
    
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def apply_rz_gate(sv: np.ndarray, theta: float, target: int, num_qubits: int):
        """Optimized RZ gate."""
        step = 1 << target
        size = 1 << num_qubits
        phase_neg = np.exp(-1j * theta / 2)
        phase_pos = np.exp(1j * theta / 2)
        
        for i in prange(size >> 1):
            idx0 = (i // step) * (step << 1) + (i % step)
            idx1 = idx0 + step
            
            sv[idx0] = sv[idx0] * phase_neg
            sv[idx1] = sv[idx1] * phase_pos
    
    
    # ===========================================================================
    # Specialized Two-Qubit Gates
    # ===========================================================================
    
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def apply_cx_gate(sv: np.ndarray, control: int, target: int, num_qubits: int):
        """Optimized CNOT gate - control qubit flips target."""
        size = 1 << num_qubits
        step_c = 1 << control
        step_t = 1 << target
        
        # Only iterate over states where control = 1
        for i in prange(size >> 1):
            # Base index with control=0
            pos1 = min(control, target)
            pos2 = max(control, target)
            step1 = 1 << pos1
            step2 = 1 << pos2
            
            base = i
            base = (base // step1) * (step1 << 1) + (base % step1)
            base = (base // step2) * (step2 << 1) + (base % step2)
            
            # Indices with control=1
            idx10 = base + step_c
            idx11 = base + step_c + step_t
            
            # Swap (controlled X on target)
            sv[idx10], sv[idx11] = sv[idx11], sv[idx10]
    
    
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def apply_cz_gate(sv: np.ndarray, control: int, target: int, num_qubits: int):
        """Optimized CZ gate - just phase flip when both qubits are |1⟩."""
        size = 1 << num_qubits
        step_c = 1 << control
        step_t = 1 << target
        
        for i in prange(size >> 2):
            pos1 = min(control, target)
            pos2 = max(control, target)
            step1 = 1 << pos1
            step2 = 1 << pos2
            
            base = i
            base = (base // step1) * (step1 << 1) + (base % step1)
            base = (base // step2) * (step2 << 1) + (base % step2)
            
            # Only negate |11⟩ component
            idx11 = base + step_c + step_t
            sv[idx11] = -sv[idx11]


def apply_optimized_gate(sv: np.ndarray, gate_name: str, qubits: list, 
                        params: list, num_qubits: int) -> np.ndarray:
    """
    Apply gate using optimized kernels when available.
    
    Returns:
        Updated statevector (in-place modification)
    """
    if not HAS_NUMBA:
        return None  # Fall back to standard implementation
    
    # Single qubit gates
    if len(qubits) == 1:
        target = qubits[0]
        
        if gate_name == 'x':
            apply_x_gate(sv, target, num_qubits)
            return sv
        elif gate_name == 'z':
            apply_z_gate(sv, target, num_qubits)
            return sv
        elif gate_name == 'h':
            apply_h_gate(sv, target, num_qubits)
            return sv
        elif gate_name == 's':
            apply_s_gate(sv, target, num_qubits)
            return sv
        elif gate_name == 't':
            apply_t_gate(sv, target, num_qubits)
            return sv
        elif gate_name == 'rx' and params:
            apply_rx_gate(sv, float(params[0]), target, num_qubits)
            return sv
        elif gate_name == 'ry' and params:
            apply_ry_gate(sv, float(params[0]), target, num_qubits)
            return sv
        elif gate_name == 'rz' and params:
            apply_rz_gate(sv, float(params[0]), target, num_qubits)
            return sv
    
    # Two qubit gates
    elif len(qubits) == 2:
        control, target = qubits[0], qubits[1]
        
        if gate_name == 'cx' or gate_name == 'cnot':
            apply_cx_gate(sv, control, target, num_qubits)
            return sv
        elif gate_name == 'cz':
            apply_cz_gate(sv, control, target, num_qubits)
            return sv
    
    # Not optimized, return None to use fallback
    return None
