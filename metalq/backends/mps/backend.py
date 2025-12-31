"""
metalq/backends/mps/backend.py - Metal Performance Shaders (MPS) Backend

Python side wrapper for the native Metal backend.
"""
from typing import Dict, List, Optional, Union, TYPE_CHECKING
import numpy as np
import time
import os
import ctypes
from dataclasses import dataclass

from ..base import Backend

if TYPE_CHECKING:
    from ...circuit import Circuit
    from ...spin import Hamiltonian


# ============================================================================
# C Interface Definitions
# ============================================================================

class MQComplex(ctypes.Structure):
    _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]

class MQGate(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),          # mq_gate_type_t enum
        ("qubits", ctypes.c_uint32 * 3),
        ("num_qubits", ctypes.c_uint32),
        ("params", ctypes.c_double * 3),
        ("num_params", ctypes.c_uint32),
    ]

class MQHamiltonian(ctypes.Structure):
    _fields_ = [
        ("num_terms", ctypes.c_uint32),
        ("num_qubits", ctypes.c_uint32),
        ("coeffs", ctypes.POINTER(ctypes.c_double)),
        ("pauli_codes", ctypes.POINTER(ctypes.c_uint8))
    ]

# Enum mapping (Must match metalq.h)
GATE_MAP = {
    'x': 0, 'y': 1, 'z': 2, 'h': 3, 's': 4, 't': 5,
    'rx': 6, 'ry': 7, 'rz': 8, 'p': 9, 'u1': 9, # u1 same as p
    'cx': 10, 'cy': 11, 'cz': 12, 'swap': 13,
    'cp': 14,
}

class MPSBackend(Backend):
    """
    GPU-accelerated backend using Apple Metal Performance Shaders.
    """
    
    def __init__(self):
        """Initialize MPS backend."""
        self._lib = self._load_library()
        if not self._lib:
            raise RuntimeError("Failed to load native MetalQ library")
        
        # Define function signatures
        self._lib.metalq_create_context.restype = ctypes.c_void_p
        self._lib.metalq_is_supported.restype = ctypes.c_bool
        
        self._lib.metalq_destroy_context.argtypes = [ctypes.c_void_p]
        self._lib.metalq_run.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(MQGate),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(MQComplex),
            ctypes.POINTER(ctypes.c_uint64)
        ]
        
        self._lib.metalq_gradient_adjoint.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(MQGate),
            ctypes.c_uint32,
            ctypes.c_void_p, # MQHamiltonian*
            ctypes.POINTER(ctypes.c_double)
        ]
        
        # Initialize native context
        self._ctx = self._lib.metalq_create_context()
        if not self._ctx:
            raise RuntimeError("Failed to create MetalQ context")
    
    def __del__(self):
        """Cleanup native resources."""
        if hasattr(self, '_lib') and self._lib and hasattr(self, '_ctx') and self._ctx:
            self._lib.metalq_destroy_context(self._ctx)
    
    def _load_library(self):
        """Load the native shared library."""
        lib_name = "libmetalq.dylib"
        # Search paths
        paths = [
            os.path.abspath(f"native/build/{lib_name}"),
            os.path.join(os.path.dirname(__file__), "../../../native/build", lib_name),
            os.path.join(os.path.dirname(__file__), "../../../../native/build", lib_name), # In case of deeper nesting
            f"/usr/local/lib/{lib_name}"
        ]
        
        for path in paths:
            if os.path.exists(path):
                try:
                    return ctypes.CDLL(path)
                except OSError:
                    continue
        
        # Fallback for installed package (if bundled)
        # ...
        
        return None

    def is_available(self) -> bool:
        """Check if Metal is supported on this device."""
        if not self._lib: return False
        return bool(self._lib.metalq_is_supported())
    
    @property
    def name(self) -> str:
        return 'mps'
    
    @property
    def max_qubits(self) -> int:
        return 30
    
    def run(self, 
            circuit: 'Circuit', 
            shots: int = 0,
            params: Optional[Union[Dict, List[float]]] = None) -> Dict:
        """Execute circuit on GPU."""
        
        # Bind parameters
        if params is not None:
            circuit = circuit.bind_parameters(params)
        
        num_qubits = circuit.num_qubits
        
        # Convert gates to C structure
        c_gates = (MQGate * len(circuit.gates))()
        
        for i, gate in enumerate(circuit.gates):
            gt = c_gates[i]
            
            # Map name to ID
            gname = gate.name.lower()
            
            if gname in GATE_MAP:
                gt.type = GATE_MAP[gname]
                # Copy params directly
                gt.num_params = len(gate.params)
                for j, p in enumerate(gate.params):
                    if j < 3: gt.params[j] = float(p)
            
            # Decompositions / Mappings
            elif gname == 'sdg':
                gt.type = GATE_MAP['p']
                gt.num_params = 1
                gt.params[0] = -np.pi/2
            elif gname == 'tdg':
                gt.type = GATE_MAP['p']
                gt.num_params = 1
                gt.params[0] = -np.pi/4
            elif gname == 'sx':
                gt.type = GATE_MAP['rx']
                gt.num_params = 1
                gt.params[0] = np.pi/2
            elif gname == 'sxdg':
                gt.type = GATE_MAP['rx']
                gt.num_params = 1
                gt.params[0] = -np.pi/2
            
            else:
                # Fallback or error? For MVP, skip or error.
                print(f"Warning: Unsupported gate {gname} for MPS, skipping/mapping to identity.")
                gt.type = 999 

            # Qubits
            gt.num_qubits = len(gate.qubits)
            for j, q in enumerate(gate.qubits):
                if j < 3: gt.qubits[j] = q

        
        # Allocate output buffers
        # Statevector: 2^n * 8 bytes (float complex)
        sv_size = 1 << num_qubits
        statevector = np.zeros(sv_size, dtype=np.complex64)
        
        # We need to pass pointer to numpy data
        sv_ptr = statevector.ctypes.data_as(ctypes.POINTER(MQComplex))
        
        # Measurement counts (if shots > 0)
        # Native API signature:
        # int metalq_run(ctx, nq, gates, ng, shots, sv_out, counts_out)
        
        # Counts logic in native is not fully implemented in verify/plan, 
        # but let's assume we get statevector back and sample in Python for MVP if shots>0
        # for maximum reliability in Phase 2/3 transition.
        # Or pass explicit None.
        
        # print(f"DEBUG: Calling metalq_run with {num_qubits} qubits, {len(circuit.gates)} gates")
        res_code = self._lib.metalq_run(
            self._ctx,
            ctypes.c_uint32(num_qubits),
            c_gates,
            ctypes.c_uint32(len(circuit.gates)),
            ctypes.c_uint32(0), # shots=0 for now in native to just get SV
            sv_ptr,
            None # counts out
        )
        # print(f"DEBUG: metalq_run returned {res_code}")
        
        if res_code != 0:
            raise RuntimeError(f"Metal execution failed with code {res_code}")
        
        result = {}
        result['statevector'] = statevector
        
        # If shots requested, sample from statevector on CPU for now (hybrid)
        # Implementing full GPU sampling is an optimization.
        if shots > 0:
            from ..cpu.measurement import sample_counts
            result['counts'] = sample_counts(statevector, shots, num_qubits)
            
        return result

    def statevector(self, circuit: 'Circuit', params=None) -> np.ndarray:
        res = self.run(circuit, shots=0, params=params)
        return res['statevector']

    def expectation(self, circuit: 'Circuit', hamiltonian: 'Hamiltonian', params=None) -> float:
        sv = self.statevector(circuit, params)
        # Calculate <psi|H|psi> on CPU
        # H |psi>
        h_psi = hamiltonian.to_matrix(circuit.num_qubits) @ sv
        return np.real(np.vdot(sv, h_psi))

    def gradient(self, circuit: 'Circuit', hamiltonian: 'Hamiltonian', params: List[float], method='parameter_shift') -> np.ndarray:
        if method != 'adjoint':
             from ..cpu.gradient import parameter_shift_gradient
             return parameter_shift_gradient(self, circuit, hamiltonian, params)

        # Native Adjoint Differentiation
        if params is not None:
            circuit = circuit.bind_parameters(params)

        # 1. Marshal Gates (Same as run)
        c_gates = (MQGate * len(circuit.gates))()
        for i, gate in enumerate(circuit.gates):
            gt = c_gates[i]
            gname = gate.name.lower()
            if gname in GATE_MAP:
                gt.type = GATE_MAP[gname]
            else:
                gt.type = 999 
            gt.num_qubits = len(gate.qubits)
            for j, q in enumerate(gate.qubits): 
                if j < 3: gt.qubits[j] = q
            gt.num_params = len(gate.params)
            for j, p in enumerate(gate.params):
                if j < 3: gt.params[j] = float(p)

        # 2. Marshal Hamiltonian
        from ...spin import PauliTerm, Hamiltonian
        if isinstance(hamiltonian, PauliTerm):
             hamiltonian = Hamiltonian([hamiltonian])
        
        terms = hamiltonian.terms
        num_terms = len(terms)
        num_qubits = circuit.num_qubits
        
        coeffs_arr = (ctypes.c_double * num_terms)()
        pauli_codes_arr = (ctypes.c_uint8 * (num_terms * num_qubits))()
        
        # Init codes to 0 (Identity)
        ctypes.memset(pauli_codes_arr, 0, ctypes.sizeof(pauli_codes_arr))
        
        for i, term in enumerate(terms):
            coeffs_arr[i] = term.coeff.real # Assume Hermitian/Real coeffs
            for p_str, q_idx in term.ops:
                code = 0
                if p_str == 'X': code = 1
                elif p_str == 'Y': code = 2
                elif p_str == 'Z': code = 3
                if q_idx < num_qubits:
                    pauli_codes_arr[i * num_qubits + q_idx] = code
        
        h_struct = MQHamiltonian()
        h_struct.num_terms = num_terms
        h_struct.num_qubits = num_qubits
        h_struct.coeffs = ctypes.cast(coeffs_arr, ctypes.POINTER(ctypes.c_double))
        h_struct.pauli_codes = ctypes.cast(pauli_codes_arr, ctypes.POINTER(ctypes.c_uint8))
        
        # 3. Output Buffer
        # Calculate total number of parameters across all gates
        total_param_count = sum(len(g.params) for g in circuit.gates)
        
        # Allocate flat gradient buffer for all parameters
        grads_out = (ctypes.c_double * total_param_count)()
        
        res = self._lib.metalq_gradient_adjoint(
            self._ctx,
            ctypes.c_uint32(num_qubits),
            c_gates,
            ctypes.c_uint32(len(circuit.gates)),
            ctypes.byref(h_struct),
            grads_out
        )
        
        if res != 0:
            raise RuntimeError(f"Native adjoint gradient failed with code {res}")
            
        # 4. Filter/Map Gradients
        # Map flat gradients back to input parameter structure.
        # We walk through gates and consume gradients from the flat buffer.
        
        final_grads = []
        grad_ptr = 0
        
        for gate in circuit.gates:
            num_params_gate = len(gate.params)
            
            if num_params_gate > 0:
                # Extract gradients for this gate
                # For v1 standard gates (RX, RY, RZ), usually 1 param.
                # If multi-param gate (e.g. U3), we extract all.
                
                # Check if this gate corresponds to trainable parameters in the input list.
                # Simplification: We assume the input `params` list strictly corresponds 
                # to the sequence of parameters encountered in circuit traversal.
                # (i.e., we return ALL computed gradients).
                
                for k in range(num_params_gate):
                    final_grads.append(grads_out[grad_ptr + k])
                
            grad_ptr += num_params_gate
            
        return np.array(final_grads)
