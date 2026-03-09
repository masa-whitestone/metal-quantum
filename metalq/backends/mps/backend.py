"""
metalq/backends/mps/backend.py - Metal Performance Shaders (MPS) Backend

Python side wrapper for the native Metal backend with enhanced error handling.
"""
from typing import Dict, List, Optional, Union, TYPE_CHECKING
import numpy as np
import time
import os
import ctypes
import math
import threading
import hashlib
import logging
from dataclasses import dataclass

from ..base import Backend
from ...exceptions import (
    ValidationError,
    InvalidParameterError,
    NativeCallError,
    NativeTimeoutError,
    LibraryLoadError,
    handle_native_result,
    validate_parameters
)

if TYPE_CHECKING:
    from ...circuit import Circuit
    from ...spin import Hamiltonian

# Configure logging
logger = logging.getLogger(__name__)


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
    Enhanced with comprehensive error handling and security features.
    """

    # Default timeout for operations (seconds)
    DEFAULT_TIMEOUT = 30.0

    def __init__(self, timeout_scale: Optional[float] = None):
        """
        Initialize MPS backend with enhanced error handling.

        Args:
            timeout_scale: Optional scale factor for operation timeouts (default 1.0)
        """
        if timeout_scale is None:
            timeout_scale = 1.0
        self.timeout_scale = timeout_scale
        self._lock = threading.Lock()
        self._lib = self._load_library()

        # Check if building documentation or running on non-macOS
        if not self._lib:
            if os.environ.get('METALQ_DOCS_BUILD') == '1' or os.environ.get('READTHEDOCS') == 'True':
                # Create a mock library for documentation build
                class MockLib:
                    def __getattr__(self, name):
                        return ctypes.CFUNCTYPE(None)
                    def metalq_create_context(self): return ctypes.c_void_p(1)
                    def metalq_is_supported(self): return True
                    def metalq_destroy_context(self, ctx): pass
                    def metalq_run(self, *args): return 0
                    def metalq_gradient_adjoint(self, *args): return 0

                self._lib = MockLib()
            else:
                raise LibraryLoadError("Failed to load native MetalQ library. Please ensure you're running on Apple Silicon with Metal support.")
        
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
        
        # Initialize native context with timeout
        try:
            self._ctx = self._call_with_timeout('metalq_create_context', timeout=5.0)
            if not self._ctx:
                raise NativeCallError("metalq_create_context", -1, "Failed to create MetalQ context")
        except Exception as e:
            logger.error(f"Failed to initialize Metal context: {e}")
            raise
    
    def __del__(self):
        """Cleanup native resources."""
        if hasattr(self, '_lib') and self._lib and hasattr(self, '_ctx') and self._ctx:
            self._lib.metalq_destroy_context(self._ctx)
    
    def _load_library(self):
        """Load the native shared library with security validation."""
        lib_name = "libmetalq.dylib"
        # Secure search paths (avoid system directories that could be compromised)
        paths = [
            os.path.abspath(f"native/build/{lib_name}"),
            os.path.join(os.path.dirname(__file__), "../../../native/build", lib_name),
            os.path.join(os.path.dirname(__file__), "../../../../native/build", lib_name),
            # Avoid /usr/local/lib for security - can be world-writable
            os.path.expanduser(f"~/.metalq/lib/{lib_name}"),
        ]

        for path in paths:
            if os.path.exists(path):
                # Validate library before loading
                if self._validate_library(path):
                    try:
                        logger.info(f"Loading library from {path}")
                        lib = ctypes.CDLL(path)
                        # Verify library has expected functions
                        self._verify_library_interface(lib)
                        return lib
                    except OSError as e:
                        logger.warning(f"Failed to load library from {path}: {e}")
                        continue

        return None

    def _validate_library(self, filepath: str) -> bool:
        """Validate library file for security."""
        try:
            # Check file permissions - should not be world-writable
            stat_info = os.stat(filepath)
            if stat_info.st_mode & 0o002:  # World-writable
                logger.warning(f"Library {filepath} is world-writable, refusing to load")
                return False

            # Log validation success
            logger.debug(f"Library validation passed for {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to validate library {filepath}: {e}")
            return False

    def _verify_library_interface(self, lib):
        """Verify that loaded library has expected functions."""
        required_functions = [
            'metalq_create_context',
            'metalq_destroy_context',
            'metalq_run',
            'metalq_gradient_adjoint',
            'metalq_is_supported'
        ]

        for func_name in required_functions:
            if not hasattr(lib, func_name):
                raise LibraryLoadError(f"Library missing required function: {func_name}")

    def _call_with_timeout(self, func_name: str, *args, timeout: Optional[float] = None, **kwargs):
        """Call native function with timeout and error handling."""
        if timeout is None:
            timeout = self.DEFAULT_TIMEOUT * self.timeout_scale

        if not hasattr(self._lib, func_name):
            raise NativeCallError(func_name, -1, f"Function {func_name} not found")

        func = getattr(self._lib, func_name)

        # Execute with timeout using threading
        result = [None]
        exception = [None]

        def execute():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            logger.error(f"Native function {func_name} timed out after {timeout}s")
            raise NativeTimeoutError(func_name, timeout)

        if exception[0]:
            logger.error(f"Native function {func_name} raised exception: {exception[0]}")
            raise exception[0]

        # Check return code if applicable
        if isinstance(result[0], int) and func_name not in ['metalq_is_supported', 'metalq_create_context']:
            handle_native_result(result[0], func_name)

        return result[0]

    def _validate_circuit(self, circuit: 'Circuit'):
        """Validate circuit before execution."""
        if circuit.num_qubits <= 0:
            raise ValidationError("Circuit must have at least 1 qubit")
        if circuit.num_qubits > self.max_qubits:
            raise ValidationError(f"Circuit has {circuit.num_qubits} qubits, maximum is {self.max_qubits}")

        if len(circuit.gates) > 100000:  # Reasonable limit
            raise ValidationError("Circuit has too many gates")

        # Validate each gate
        for i, gate in enumerate(circuit.gates):
            if not gate.name:
                raise ValidationError(f"Gate {i} has no name")

            # Check qubit indices
            for qubit in gate.qubits:
                if qubit < 0 or qubit >= circuit.num_qubits:
                    raise ValidationError(f"Gate {i} ({gate.name}) has invalid qubit index {qubit}")

            # Validate parameters
            for j, param in enumerate(gate.params):
                if isinstance(param, (int, float)):
                    if math.isnan(param):
                        raise ValidationError(f"Gate {i} ({gate.name}) parameter {j} is NaN")
                    if math.isinf(param):
                        raise ValidationError(f"Gate {i} ({gate.name}) parameter {j} is infinite")
                    if abs(param) > 1000:
                        raise ValidationError(f"Gate {i} ({gate.name}) parameter {j} value {param} exceeds reasonable bounds")

    def _calculate_timeout(self, num_qubits: int, num_gates: int) -> float:
        """Calculate appropriate timeout based on circuit complexity."""
        base = 10.0
        complexity_factor = (2 ** (num_qubits / 10)) * (num_gates / 100)
        timeout = base + complexity_factor * self.timeout_scale
        max_timeout = 300.0 * self.timeout_scale  # 5 minutes max
        return min(timeout, max_timeout)

    def is_available(self) -> bool:
        """Check if Metal is supported on this device."""
        if not self._lib: return False
        try:
            return bool(self._call_with_timeout('metalq_is_supported', timeout=2.0))
        except Exception:
            return False
    
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
        """Execute circuit on GPU with enhanced error handling."""
        start_time = time.perf_counter()

        # Validate inputs
        if shots < 0:
            raise ValidationError("Shots must be non-negative")
        if shots > 1000000:  # Reasonable limit
            raise ValidationError("Shots exceeds maximum (1000000)")

        # Bind parameters
        if params is not None:
            try:
                circuit = circuit.bind_parameters(params)
            except Exception as e:
                raise InvalidParameterError(f"Failed to bind parameters: {e}")

        # Validate circuit
        self._validate_circuit(circuit)

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

        # Check memory requirements
        memory_required = sv_size * 8  # complex64 = 8 bytes
        if memory_required > 16 * 1024**3:  # 16GB limit
            raise ValidationError(
                f"Circuit requires {memory_required / 1024**3:.1f}GB, exceeding memory limit"
            )

        try:
            statevector = np.zeros(sv_size, dtype=np.complex64)
        except MemoryError:
            raise ValidationError(f"Failed to allocate memory for {num_qubits} qubits")

        # We need to pass pointer to numpy data
        sv_ptr = statevector.ctypes.data_as(ctypes.POINTER(MQComplex))

        # Calculate appropriate timeout
        num_gates = len(circuit.gates)
        timeout = self._calculate_timeout(num_qubits, num_gates)

        logger.info(f"Executing circuit: {num_qubits} qubits, {num_gates} gates, timeout={timeout}s")

        # Execute circuit with timeout and error handling
        try:
            res_code = self._call_with_timeout(
                'metalq_run',
                self._ctx,
                ctypes.c_uint32(num_qubits),
                c_gates,
                ctypes.c_uint32(len(circuit.gates)),
                ctypes.c_uint32(0),  # shots=0 for now in native to just get SV
                sv_ptr,
                None,  # counts out
                timeout=timeout
            )
        except Exception as e:
            logger.error(f"Circuit execution failed: {e}")
            raise
        
        result = {
            'statevector': statevector,
            'time_ms': (time.perf_counter() - start_time) * 1000
        }

        # If shots requested, sample from statevector on CPU for now (hybrid)
        # Implementing full GPU sampling is an optimization.
        if shots > 0:
            try:
                from ..cpu.measurement import sample_counts
                result['counts'] = sample_counts(statevector, shots, num_qubits)
            except Exception as e:
                logger.error(f"Sampling failed: {e}")
                raise NativeCallError('sampling', -1, f"Failed to sample: {e}")

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
        
        # Calculate timeout for gradient computation
        timeout = self._calculate_timeout(num_qubits, len(circuit.gates)) * 2  # Gradient is more expensive

        logger.info(f"Computing gradient with timeout={timeout}s")

        # Compute gradient with timeout
        try:
            res = self._call_with_timeout(
                'metalq_gradient_adjoint',
                self._ctx,
                ctypes.c_uint32(num_qubits),
                c_gates,
                ctypes.c_uint32(len(circuit.gates)),
                ctypes.byref(h_struct),
                grads_out,
                timeout=timeout
            )
        except Exception as e:
            logger.error(f"Gradient computation failed: {e}")
            raise
            
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
