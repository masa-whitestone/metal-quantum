"""
metalq/backends/cpu/statevector.py - NumPy State Vector Operations

NumPy を使用した状態ベクトル操作。Numba がない場合のフォールバックとしても機能。
"""
import numpy as np
from typing import List, Dict, Any, Union, Optional

def initialize_statevector(num_qubits: int) -> np.ndarray:
    """
    Initialize statevector to |0...0⟩.
    
    Args:
        num_qubits: Number of qubits
    
    Returns:
        Complex statevector of size 2^n
    """
    size = 1 << num_qubits
    sv = np.zeros(size, dtype=np.complex128)
    sv[0] = 1.0
    return sv

def apply_gates(sv: np.ndarray, gates: List, num_qubits: int) -> np.ndarray:
    """
    Apply gates sequentially using NumPy (matrix multiplication/tensor contraction).
    
    Args:
        sv: Statevector
        gates: List of gates
        num_qubits: Number of qubits
    
    Returns:
        Updated statevector
    """
    for gate in gates:
        sv = apply_gate(sv, gate, num_qubits)
    return sv

def apply_gate(sv: np.ndarray, gate: Any, num_qubits: int) -> np.ndarray:
    """Apply a single gate to the statevector."""
    # Ensure sv is reshaped to (2, 2, ..., 2) for easy tensor operations
    sv_reshaped = sv.reshape([2] * num_qubits)
    
    name = gate.name if hasattr(gate, 'name') else gate['name']
    qubits = gate.qubits if hasattr(gate, 'qubits') else gate['qubits']
    params = gate.params if hasattr(gate, 'params') else gate.get('params', [])

    if name == 'barrier':
        return sv

    matrix = get_gate_matrix(name, params)
    
    # Calculate axes for tensordot
    # We want to contract the gate matrix with the specific qubit axes of the statevector
    # Gate matrix shape: (2^k, 2^k) where k is len(qubits)
    # Reshape gate matrix to (2, 2, ..., 2, 2, ..., 2) (k times output, k times input)
    k = len(qubits)
    matrix_reshaped = matrix.reshape([2] * (2 * k))
    
    # Input axes of the gate correspond to the last k dimensions of matrix_reshaped
    # We want to contract these with the corresponding qubit axes in sv
    # einsum is often easier for this
    
    # Construct einsum string
    # E.g. 1 qubit gate on qubit 0 (n=2): 'ab,b' -> 'a' is wrong.
    # sv indices: i_0, i_1, ..., i_{n-1}
    # gate indices: out_q, in_q
    
    # Let's map ascii characters to indices
    # Statevector indices
    all_indices = [chr(97 + i) for i in range(num_qubits)] # a, b, c...
    
    # Gate indices
    # Output indices for the target qubits will be new characters
    # Input indices for the target qubits will be the existing characters from sv
    target_indices = [all_indices[q] for q in qubits]
    new_indices = [chr(97 + num_qubits + i) for i in range(len(qubits))] # New chars
    
    # Gate matrix indices: [out_0, ..., out_{k-1}, in_0, ..., in_{k-1}]
    gate_indices_str = "".join(new_indices) + "".join(target_indices)
    
    # Input statevector indices
    input_str = "".join(all_indices)
    
    # Output statevector indices: replace target indices with new indices
    output_indices = list(all_indices)
    for i, q in enumerate(qubits):
        output_indices[q] = new_indices[i]
    output_str = "".join(output_indices)
    
    # Einsum equation
    equation = f"{gate_indices_str},{input_str}->{output_str}"
    
    # Apply
    new_sv = np.einsum(equation, matrix_reshaped, sv_reshaped)
    
    return new_sv.flatten()

# ============================================================================
# Gate Matrices
# ============================================================================

# Pre-computed gate matrices (cache)
_GATE_CACHE = {}

def get_gate_matrix(name: str, params: List = None) -> np.ndarray:
    """Get gate matrix, with caching for non-parameterized gates."""
    params = params or []
    
    # Check cache for non-parameterized gates
    if not params and name in _GATE_CACHE:
        return _GATE_CACHE[name]
    
    matrix = _compute_gate_matrix(name, params)
    
    # Cache non-parameterized gates
    if not params:
        _GATE_CACHE[name] = matrix
    
    return matrix


def _compute_gate_matrix(name: str, params: List) -> np.ndarray:
    """Compute gate matrix."""
    
    # ==================== Single-Qubit Gates ====================
    
    if name == 'id' or name == 'i':
        return np.eye(2, dtype=np.complex128)
    
    if name == 'x':
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    
    if name == 'y':
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    
    if name == 'z':
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    if name == 'h':
        return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    
    if name == 's':
        return np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    
    if name == 'sdg':
        return np.array([[1, 0], [0, -1j]], dtype=np.complex128)
    
    if name == 't':
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
    
    if name == 'tdg':
        return np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128)
    
    if name == 'sx':
        return np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=np.complex128) / 2
    
    if name == 'sxdg':
        return np.array([[1-1j, 1+1j], [1+1j, 1-1j]], dtype=np.complex128) / 2
    
    # ==================== Rotation Gates ====================
    
    if name == 'rx':
        theta = float(params[0])
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    
    if name == 'ry':
        theta = float(params[0])
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    
    if name == 'rz':
        theta = float(params[0])
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=np.complex128)
    
    if name == 'p' or name == 'u1':
        lam = float(params[0])
        return np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=np.complex128)
    
    if name == 'u2':
        phi, lam = float(params[0]), float(params[1])
        return np.array([
            [1, -np.exp(1j * lam)],
            [np.exp(1j * phi), np.exp(1j * (phi + lam))]
        ], dtype=np.complex128) / np.sqrt(2)
    
    if name == 'u' or name == 'u3':
        theta, phi, lam = float(params[0]), float(params[1]), float(params[2])
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
        ], dtype=np.complex128)
    
    if name == 'r':
        theta, phi = float(params[0]), float(params[1])
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [c, -1j * np.exp(-1j * phi) * s],
            [-1j * np.exp(1j * phi) * s, c]
        ], dtype=np.complex128)
    
    # ==================== Two-Qubit Gates ====================
    
    if name == 'cx' or name == 'cnot':
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.complex128)
    
    if name == 'cy':
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1j, 0]
        ], dtype=np.complex128)
    
    if name == 'cz':
        return np.diag([1, 1, 1, -1]).astype(np.complex128)
    
    if name == 'ch':
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)],
            [0, 0, 1/np.sqrt(2), -1/np.sqrt(2)]
        ], dtype=np.complex128)
    
    if name == 'swap':
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.complex128)
    
    if name == 'iswap':
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.complex128)
    
    if name == 'crx':
        theta = float(params[0])
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, -1j * s],
            [0, 0, -1j * s, c]
        ], dtype=np.complex128)
    
    if name == 'cry':
        theta = float(params[0])
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, -s],
            [0, 0, s, c]
        ], dtype=np.complex128)
    
    if name == 'crz':
        theta = float(params[0])
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.exp(-1j * theta / 2), 0],
            [0, 0, 0, np.exp(1j * theta / 2)]
        ], dtype=np.complex128)
    
    if name == 'cp' or name == 'cu1':
        lam = float(params[0])
        return np.diag([1, 1, 1, np.exp(1j * lam)]).astype(np.complex128)
    
    if name == 'rxx':
        theta = float(params[0])
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [c, 0, 0, -1j * s],
            [0, c, -1j * s, 0],
            [0, -1j * s, c, 0],
            [-1j * s, 0, 0, c]
        ], dtype=np.complex128)
    
    if name == 'ryy':
        theta = float(params[0])
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [c, 0, 0, 1j * s],
            [0, c, -1j * s, 0],
            [0, -1j * s, c, 0],
            [1j * s, 0, 0, c]
        ], dtype=np.complex128)
    
    if name == 'rzz':
        theta = float(params[0])
        return np.diag([
            np.exp(-1j * theta / 2),
            np.exp(1j * theta / 2),
            np.exp(1j * theta / 2),
            np.exp(-1j * theta / 2)
        ]).astype(np.complex128)
    
    if name == 'rzx':
        theta = float(params[0])
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [c, -1j * s, 0, 0],
            [-1j * s, c, 0, 0],
            [0, 0, c, 1j * s],
            [0, 0, 1j * s, c]
        ], dtype=np.complex128)
    
    # ==================== Three-Qubit Gates ====================
    
    if name == 'ccx' or name == 'toffoli':
        m = np.eye(8, dtype=np.complex128)
        m[6, 6], m[6, 7] = 0, 1
        m[7, 6], m[7, 7] = 1, 0
        return m
    
    if name == 'ccz':
        return np.diag([1, 1, 1, 1, 1, 1, 1, -1]).astype(np.complex128)
    
    if name == 'cswap' or name == 'fredkin':
        m = np.eye(8, dtype=np.complex128)
        m[5, 5], m[5, 6] = 0, 1
        m[6, 5], m[6, 6] = 1, 0
        return m
    
    raise ValueError(f"Unknown gate: {name}")
