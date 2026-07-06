"""
metalq/backends/mps/backend.py - Metal Performance Shaders (MPS) Backend

Python side wrapper for the native Metal backend with enhanced error handling.
"""
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
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

# Enum mapping (MUST match the mq_gate_type_t enum order in native/include/metalq.h)
GATE_MAP = {
    'x': 0, 'y': 1, 'z': 2, 'h': 3, 's': 4, 't': 5,
    'rx': 6, 'ry': 7, 'rz': 8, 'p': 9, 'u1': 9, # u1 same as p
    'cx': 10, 'cy': 11, 'cz': 12, 'swap': 13,
    'cp': 14,
    'ccx': 16, 'cswap': 17, 'ccz': 18,
    'u3': 19, 'u': 19,  # universal single-qubit gate U(theta, phi, lam)
    'ch': 20,           # controlled-Hadamard
}

# Gates whose parameter is natively differentiable by the adjoint kernel.
# Any parameterized gate outside this set forces a parameter-shift fallback.
ADJOINT_DIFFERENTIABLE = frozenset({'rx', 'ry', 'rz', 'p', 'u1', 'cp'})

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
        self._ctx = None
        self._ctx_poisoned = False
        self._ctx_poisoned_reason = None
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

        # GPU expectation (optional: older dylibs may not export it)
        if hasattr(self._lib, 'metalq_expectation'):
            self._lib.metalq_expectation.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(MQGate),
                ctypes.c_uint32,
                ctypes.c_void_p,  # MQHamiltonian*
                ctypes.POINTER(ctypes.c_double)
            ]
            self._lib.metalq_expectation.restype = ctypes.c_int

        # GPU-resident sampling (optional: older dylibs may not export it;
        # run() falls back to CPU sampling from the statevector otherwise).
        if hasattr(self._lib, 'metalq_sample'):
            self._lib.metalq_sample.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(MQGate),
                ctypes.c_uint32,
                ctypes.c_uint32,  # shots
                ctypes.c_uint64,  # seed
                ctypes.POINTER(ctypes.c_uint32),  # out_samples
            ]
            self._lib.metalq_sample.restype = ctypes.c_int

        # Fused adjoint gradient + energy (optional: older dylibs may not
        # export it; expectation_and_gradient() falls back to separate
        # expectation() + gradient() calls in that case).
        if hasattr(self._lib, 'metalq_gradient_adjoint_energy'):
            self._lib.metalq_gradient_adjoint_energy.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(MQGate),
                ctypes.c_uint32,
                ctypes.c_void_p,  # MQHamiltonian*
                ctypes.POINTER(ctypes.c_double),  # out_gradients
                ctypes.POINTER(ctypes.c_double),  # out_energy
            ]
            self._lib.metalq_gradient_adjoint_energy.restype = ctypes.c_int

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
            if getattr(self, '_ctx_poisoned', False):
                logger.warning(
                    "Skipping MetalQ context destruction because a timed-out native "
                    "call may still be using the context"
                )
                return
            self._lib.metalq_destroy_context(self._ctx)

    def _mark_context_unusable(self, func_name: str, timeout: float):
        """Poison the backend context after an unsafe native timeout."""
        self._ctx_poisoned = True
        self._ctx_poisoned_reason = (
            f"Native function {func_name} timed out after {timeout}s; "
            "the backend context may still be in use on a background thread"
        )
        self._ctx = None
        logger.error(self._ctx_poisoned_reason)

    def _ensure_context_usable(self, func_name: str):
        """Fail closed after a timed-out context-backed call."""
        if self._ctx_poisoned:
            raise NativeCallError(
                func_name,
                -1,
                self._ctx_poisoned_reason or "MetalQ backend context is unusable"
            )

    def _load_library(self):
        """Load the native shared library with security validation."""
        lib_name = "libmetalq.dylib"
        # Secure search paths (avoid system directories that could be compromised).
        # Dev-tree builds (native/build) take priority over the packaged copy so
        # a fresh `make` always wins during development.
        paths = [
            os.path.abspath(f"native/build/{lib_name}"),
            os.path.join(os.path.dirname(__file__), "../../../native/build", lib_name),
            os.path.join(os.path.dirname(__file__), "../../../../native/build", lib_name),
            # Wheel/editable installs: binaries shipped inside the package
            # (metalq/lib, populated by `make` — default.metallib sits next to
            # the dylib, where the native dladdr lookup finds it).
            os.path.join(os.path.dirname(__file__), "../../lib", lib_name),
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

        if func_name in {'metalq_run', 'metalq_gradient_adjoint', 'metalq_gradient_adjoint_energy', 'metalq_sample', 'metalq_destroy_context'}:
            self._ensure_context_usable(func_name)

        func = getattr(self._lib, func_name)

        with self._lock:
            # Re-check after waiting for the lock to avoid racing a prior timeout.
            if func_name in {'metalq_run', 'metalq_gradient_adjoint', 'metalq_gradient_adjoint_energy', 'metalq_sample', 'metalq_destroy_context'}:
                self._ensure_context_usable(func_name)

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
                if func_name in {'metalq_run', 'metalq_gradient_adjoint', 'metalq_gradient_adjoint_energy', 'metalq_sample', 'metalq_destroy_context'}:
                    self._mark_context_unusable(func_name, timeout)
                else:
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
    
    def _marshal_gates(self, circuit: 'Circuit'):
        """Convert circuit gates to the C gate array (shared by run/expectation/gradient).

        Fails closed: any gate name that cannot be mapped to a native gate
        raises ValidationError rather than silently applying identity. No-op
        gates ('id', 'barrier') are dropped from the marshaled array.
        """
        marshaled = []

        for i, gate in enumerate(circuit.gates):
            gname = gate.name.lower()

            # No-op gates: drop them (identity / visual-only). They carry no
            # parameters, so dropping them does not disturb gradient bookkeeping.
            if gname in ('id', 'i', 'barrier'):
                continue

            gt = MQGate()

            if gname == 'u2':
                # U2(phi, lam) == U3(pi/2, phi, lam)
                phi, lam = float(gate.params[0]), float(gate.params[1])
                gt.type = GATE_MAP['u3']
                gt.num_params = 3
                gt.params[0] = np.pi / 2
                gt.params[1] = phi
                gt.params[2] = lam
            elif gname == 'r':
                # R(theta, phi) == U3(theta, phi - pi/2, pi/2 - phi) (exact).
                theta, phi = float(gate.params[0]), float(gate.params[1])
                gt.type = GATE_MAP['u3']
                gt.num_params = 3
                gt.params[0] = theta
                gt.params[1] = phi - np.pi / 2
                gt.params[2] = np.pi / 2 - phi
            elif gname == 'cs':
                # Controlled-S == CP(pi/2)
                gt.type = GATE_MAP['cp']
                gt.num_params = 1
                gt.params[0] = np.pi / 2
            elif gname == 'csdg':
                # Controlled-S-dagger == CP(-pi/2)
                gt.type = GATE_MAP['cp']
                gt.num_params = 1
                gt.params[0] = -np.pi / 2
            elif gname in GATE_MAP:
                gt.type = GATE_MAP[gname]
                gt.num_params = len(gate.params)
                for j, p in enumerate(gate.params):
                    if j < 3:
                        gt.params[j] = float(p)

            # Decompositions / Mappings
            elif gname == 'sdg':
                gt.type = GATE_MAP['p']
                gt.num_params = 1
                gt.params[0] = -np.pi / 2
            elif gname == 'tdg':
                gt.type = GATE_MAP['p']
                gt.num_params = 1
                gt.params[0] = -np.pi / 4
            elif gname == 'sx':
                # sx == RX(pi/2) up to a global phase (accepted, matches sxdg).
                gt.type = GATE_MAP['rx']
                gt.num_params = 1
                gt.params[0] = np.pi / 2
            elif gname == 'sxdg':
                gt.type = GATE_MAP['rx']
                gt.num_params = 1
                gt.params[0] = -np.pi / 2

            else:
                # Fail closed: never fall through to a silent identity.
                raise ValidationError(
                    f"Gate '{gate.name}' (index {i}) is not supported by the MPS "
                    f"(GPU) backend. Use the CPU backend, or decompose it into "
                    f"supported gates."
                )

            gt.num_qubits = len(gate.qubits)
            for j, q in enumerate(gate.qubits):
                if j < 3:
                    gt.qubits[j] = q

            marshaled.append(gt)

        c_gates = (MQGate * len(marshaled))()
        for i, gt in enumerate(marshaled):
            c_gates[i] = gt
        return c_gates

    def _marshal_hamiltonian(self, hamiltonian: 'Hamiltonian', num_qubits: int):
        """Convert a Hamiltonian to the C struct.

        Returns (h_struct, keepalive) — keepalive holds the ctypes arrays the
        struct points into; keep it referenced for the duration of the call.
        """
        terms = hamiltonian.terms
        num_terms = len(terms)

        coeffs_arr = (ctypes.c_double * num_terms)()
        pauli_codes_arr = (ctypes.c_uint8 * (num_terms * num_qubits))()
        ctypes.memset(pauli_codes_arr, 0, ctypes.sizeof(pauli_codes_arr))

        for i, term in enumerate(terms):
            coeff = complex(term.coeff)
            # Fail closed on non-Hermitian input: the native path only carries the
            # real coefficient, so a non-negligible imaginary part would be
            # silently dropped, producing a wrong (or non-real) expectation value.
            if abs(coeff.imag) > 1e-10:
                raise ValidationError(
                    f"Hamiltonian term {i} has a non-negligible imaginary "
                    f"coefficient ({coeff.imag:g}); the MPS backend only supports "
                    f"Hermitian (real-coefficient) observables."
                )
            coeffs_arr[i] = coeff.real
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

        return h_struct, (coeffs_arr, pauli_codes_arr)

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
        c_gates = self._marshal_gates(circuit)

        # Statevector: 2^n * 8 bytes (float complex)
        sv_size = 1 << num_qubits

        # Check memory requirements. The GPU-sampling path never materializes the
        # statevector on the host, but the GPU still allocates 2^n amplitudes, so
        # the limit applies to both paths.
        memory_required = sv_size * 8  # complex64 = 8 bytes
        if memory_required > 16 * 1024**3:  # 16GB limit
            raise ValidationError(
                f"Circuit requires {memory_required / 1024**3:.1f}GB, exceeding memory limit"
            )

        # Calculate appropriate timeout (num_gates is the marshaled count, which
        # may be smaller than len(circuit.gates) after dropping no-op gates).
        num_gates = len(c_gates)
        timeout = self._calculate_timeout(num_qubits, num_gates)

        # GPU-resident sampling: when only counts are wanted, sample directly on
        # the GPU so the 2^n statevector never gets copied back to Python.
        if shots > 0 and hasattr(self._lib, 'metalq_sample'):
            logger.info(
                f"Sampling on GPU: {num_qubits} qubits, {num_gates} gates, "
                f"{shots} shots, timeout={timeout}s"
            )
            # Draw the seed from numpy's global RNG so np.random.seed(...)
            # reproducibility semantics keep working through to the native PRNG.
            seed = int(np.random.randint(0, 2**63 - 1))
            out_samples = (ctypes.c_uint32 * shots)()
            try:
                self._call_with_timeout(
                    'metalq_sample',
                    self._ctx,
                    ctypes.c_uint32(num_qubits),
                    c_gates,
                    ctypes.c_uint32(len(c_gates)),
                    ctypes.c_uint32(shots),
                    ctypes.c_uint64(seed),
                    out_samples,
                    timeout=timeout
                )
            except Exception as e:
                logger.error(f"GPU sampling failed: {e}")
                raise
            indices = np.ctypeslib.as_array(out_samples)
            # 'statevector' is intentionally omitted: the whole point of GPU
            # sampling is to avoid the full readback.
            return {
                'counts': self._counts_from_samples(indices, num_qubits),
                'time_ms': (time.perf_counter() - start_time) * 1000
            }

        # Statevector path (shots == 0, or an old dylib without metalq_sample).
        try:
            statevector = np.zeros(sv_size, dtype=np.complex64)
        except MemoryError:
            raise ValidationError(f"Failed to allocate memory for {num_qubits} qubits")

        # We need to pass pointer to numpy data
        sv_ptr = statevector.ctypes.data_as(ctypes.POINTER(MQComplex))

        logger.info(f"Executing circuit: {num_qubits} qubits, {num_gates} gates, timeout={timeout}s")

        # Execute circuit with timeout and error handling
        try:
            res_code = self._call_with_timeout(
                'metalq_run',
                self._ctx,
                ctypes.c_uint32(num_qubits),
                c_gates,
                ctypes.c_uint32(len(c_gates)),
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

        # Fallback CPU sampling (old dylib without metalq_sample): sample from
        # the statevector we just read back.
        if shots > 0:
            try:
                from ..cpu.measurement import sample_counts
                result['counts'] = sample_counts(statevector, shots, num_qubits)
            except Exception as e:
                logger.error(f"Sampling failed: {e}")
                raise NativeCallError('sampling', -1, f"Failed to sample: {e}")

        return result

    @staticmethod
    def _counts_from_samples(indices: np.ndarray, num_qubits: int) -> Dict[str, int]:
        """Aggregate sampled basis-state indices into a counts dict.

        Key format is identical to cpu.measurement.sample_counts:
        ``format(idx, f'0{num_qubits}b')`` (big-endian bitstring, qubit 0 = LSB
        = rightmost character). np.unique keeps the cost O(shots) rather than
        O(2^n), so it stays cheap at high qubit counts.
        """
        uniq, cnts = np.unique(indices, return_counts=True)
        return {
            format(int(i), f'0{num_qubits}b'): int(c)
            for i, c in zip(uniq, cnts)
        }

    def statevector(self, circuit: 'Circuit', params=None) -> np.ndarray:
        res = self.run(circuit, shots=0, params=params)
        return res['statevector']

    def expectation(self, circuit: 'Circuit', hamiltonian: 'Hamiltonian', params=None) -> float:
        from ...spin import PauliTerm, Hamiltonian
        if isinstance(hamiltonian, PauliTerm):
            hamiltonian = Hamiltonian([hamiltonian])

        if params is not None:
            try:
                circuit = circuit.bind_parameters(params)
            except Exception as e:
                raise InvalidParameterError(f"Failed to bind parameters: {e}")

        # Native GPU path: circuit + Pauli-term reduction stay on GPU,
        # no statevector readback and no dense Hamiltonian matrix.
        if hasattr(self._lib, 'metalq_expectation'):
            self._validate_circuit(circuit)
            num_qubits = circuit.num_qubits
            c_gates = self._marshal_gates(circuit)
            h_struct, _keepalive = self._marshal_hamiltonian(hamiltonian, num_qubits)

            out_value = ctypes.c_double(0.0)
            num_terms = max(1, h_struct.num_terms)
            timeout = self._calculate_timeout(num_qubits, len(c_gates)) \
                * max(2.0, num_terms / 10.0)

            self._call_with_timeout(
                'metalq_expectation',
                self._ctx,
                ctypes.c_uint32(num_qubits),
                c_gates,
                ctypes.c_uint32(len(c_gates)),
                ctypes.byref(h_struct),
                ctypes.byref(out_value),
                timeout=timeout
            )
            return float(out_value.value)

        # Fallback (old dylib): read back statevector, contract term-wise
        # without materializing the dense 2^n x 2^n Hamiltonian matrix.
        sv = self.statevector(circuit)
        return self._expectation_from_statevector(sv, hamiltonian, circuit.num_qubits)

    @staticmethod
    def _expectation_from_statevector(sv: np.ndarray, hamiltonian: 'Hamiltonian',
                                      num_qubits: int) -> float:
        """Matrix-free <psi|H|psi> (shared with the CPU backend)."""
        from ..cpu.statevector import expectation_from_statevector
        return expectation_from_statevector(sv, hamiltonian, num_qubits)

    def gradient(self, circuit: 'Circuit', hamiltonian: 'Hamiltonian', params: List[float], method='parameter_shift') -> np.ndarray:
        if method != 'adjoint':
             from ..cpu.gradient import parameter_shift_gradient
             return parameter_shift_gradient(self, circuit, hamiltonian, params)

        # The native adjoint kernel only differentiates rx/ry/rz/p/u1/cp. If any
        # gate that CARRIES a Parameter is outside that set (e.g. a parameterized
        # u3), fall back to parameter-shift for the whole circuit. Constant-valued
        # gates of any supported type are fine (they are not differentiated).
        from ...parameter import is_parameterized
        for g in circuit.gates:
            if g.name.lower() not in ADJOINT_DIFFERENTIABLE and \
                    any(is_parameterized(p) for p in g.params):
                if not getattr(self, '_adjoint_fallback_warned', False):
                    logger.warning(
                        "Adjoint gradient not implemented for parameterized gate "
                        f"'{g.name}'; falling back to parameter-shift gradient."
                    )
                    self._adjoint_fallback_warned = True
                from ..cpu.gradient import parameter_shift_gradient
                return parameter_shift_gradient(self, circuit, hamiltonian, params)

        # Native Adjoint Differentiation
        if params is not None:
            circuit = circuit.bind_parameters(params)

        # 1. Marshal Gates (Same as run)
        c_gates = self._marshal_gates(circuit)

        # 2. Marshal Hamiltonian
        from ...spin import PauliTerm, Hamiltonian
        if isinstance(hamiltonian, PauliTerm):
             hamiltonian = Hamiltonian([hamiltonian])

        num_qubits = circuit.num_qubits
        h_struct, _keepalive = self._marshal_hamiltonian(hamiltonian, num_qubits)

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
                ctypes.c_uint32(len(c_gates)),
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

    def expectation_and_gradient(self, circuit: 'Circuit', hamiltonian: 'Hamiltonian',
                                  params: List[float]) -> Tuple[float, np.ndarray]:
        """
        Compute <H> and its gradient in ONE fused GPU pass via the native
        adjoint kernel (metalq_gradient_adjoint_energy) instead of running the
        circuit forward twice (once for expectation(), once for gradient()).

        Falls back to separate expectation() + gradient(method='adjoint')
        calls (which itself falls back further to parameter-shift, with the
        usual once-only warning) if:
          - the circuit contains a parameter-carrying gate outside the
            natively adjoint-differentiable set (same pre-scan as
            gradient(method='adjoint')), or
          - the loaded dylib predates metalq_gradient_adjoint_energy.

        Returns:
            Tuple of the energy and the gradient array of shape (num_params,).
        """
        from ...parameter import is_parameterized
        needs_fallback = any(
            g.name.lower() not in ADJOINT_DIFFERENTIABLE and
            any(is_parameterized(p) for p in g.params)
            for g in circuit.gates
        )

        if needs_fallback or not hasattr(self._lib, 'metalq_gradient_adjoint_energy'):
            energy = self.expectation(circuit, hamiltonian, params)
            grad = self.gradient(circuit, hamiltonian, params, method='adjoint')
            return energy, grad

        # Native fused Adjoint Differentiation + Energy
        if params is not None:
            circuit = circuit.bind_parameters(params)

        # 1. Marshal Gates (Same as run/gradient)
        c_gates = self._marshal_gates(circuit)

        # 2. Marshal Hamiltonian
        from ...spin import PauliTerm, Hamiltonian
        if isinstance(hamiltonian, PauliTerm):
            hamiltonian = Hamiltonian([hamiltonian])

        num_qubits = circuit.num_qubits
        h_struct, _keepalive = self._marshal_hamiltonian(hamiltonian, num_qubits)

        # 3. Output buffers
        total_param_count = sum(len(g.params) for g in circuit.gates)
        grads_out = (ctypes.c_double * total_param_count)()
        energy_out = ctypes.c_double(0.0)

        # Gradient (adjoint) is the more expensive of the two; the fused call
        # does no more GPU work than gradient() alone, so reuse its timeout.
        timeout = self._calculate_timeout(num_qubits, len(circuit.gates)) * 2

        logger.info(f"Computing fused expectation+gradient with timeout={timeout}s")

        try:
            self._call_with_timeout(
                'metalq_gradient_adjoint_energy',
                self._ctx,
                ctypes.c_uint32(num_qubits),
                c_gates,
                ctypes.c_uint32(len(c_gates)),
                ctypes.byref(h_struct),
                grads_out,
                ctypes.byref(energy_out),
                timeout=timeout
            )
        except Exception as e:
            logger.error(f"Fused expectation+gradient computation failed: {e}")
            raise

        # 4. Filter/Map Gradients (same mapping as gradient())
        final_grads = []
        grad_ptr = 0

        for gate in circuit.gates:
            num_params_gate = len(gate.params)
            if num_params_gate > 0:
                for k in range(num_params_gate):
                    final_grads.append(grads_out[grad_ptr + k])
            grad_ptr += num_params_gate

        return float(energy_out.value), np.array(final_grads)
