"""
Secure MPS Backend with Enhanced Error Handling

Drop-in replacement for the original MPS backend with:
- Comprehensive error handling
- Timeout mechanisms
- Input validation
- Security logging
"""
from typing import Dict, List, Optional, Union, TYPE_CHECKING
import numpy as np
import time
import os
import ctypes
import logging
from dataclasses import dataclass

from ..base import Backend
from .backend import MQComplex, MQGate, MQHamiltonian, GATE_MAP
from .native_wrapper import create_secure_native_wrapper, NativeCallWrapper
from ...exceptions import (
    ValidationError,
    InvalidParameterError,
    GPUError,
    NativeCallError,
    validate_parameters
)

if TYPE_CHECKING:
    from ...circuit import Circuit
    from ...spin import Hamiltonian

# Configure logging
logger = logging.getLogger(__name__)


class SecureMPSBackend(Backend):
    """
    Secure GPU-accelerated backend using Apple Metal Performance Shaders.

    This implementation adds comprehensive error handling, timeouts,
    and input validation to the original MPS backend.
    """

    def __init__(self, timeout_scale: float = 1.0):
        """
        Initialize Secure MPS backend.

        Args:
            timeout_scale: Scale factor for operation timeouts (default 1.0)
        """
        self.timeout_scale = timeout_scale
        self._wrapper: Optional[NativeCallWrapper] = None
        self._ctx = None

        # Initialize the secure wrapper
        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the native library with secure wrapper."""
        try:
            # Create secure wrapper with custom timeout
            self._wrapper = create_secure_native_wrapper(
                library_name='libmetalq.dylib',
                default_timeout=30.0 * self.timeout_scale
            )

            # Check if Metal is supported
            if not self._wrapper.is_supported():
                raise GPUError("Metal is not supported on this device")

            # Create context with timeout
            self._ctx = self._wrapper.create_context(timeout=5.0)
            if not self._ctx:
                raise GPUError("Failed to create Metal context")

            logger.info("Secure MPS backend initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MPS backend: {e}")
            # Fallback message for users
            raise GPUError(
                "MPS backend initialization failed. "
                "Please ensure you're running on Apple Silicon with Metal support."
            )

    def __del__(self):
        """Cleanup native resources."""
        if self._wrapper and self._ctx:
            try:
                self._wrapper.destroy_context(self._ctx, timeout=5.0)
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

    def is_available(self) -> bool:
        """Check if Metal is supported on this device."""
        return self._wrapper is not None and self._wrapper.is_supported()

    @property
    def name(self) -> str:
        return 'secure_mps'

    @property
    def max_qubits(self) -> int:
        # MPS uses half precision, limiting memory
        return 30

    def _validate_circuit(self, circuit: 'Circuit'):
        """
        Validate circuit before execution.

        Args:
            circuit: Circuit to validate

        Raises:
            ValidationError: If circuit is invalid
        """
        # Check qubit count
        if circuit.num_qubits <= 0:
            raise ValidationError("Circuit must have at least 1 qubit")
        if circuit.num_qubits > self.max_qubits:
            raise ValidationError(
                f"Circuit has {circuit.num_qubits} qubits, "
                f"maximum is {self.max_qubits}"
            )

        # Check gates
        if len(circuit.gates) > 100000:  # Reasonable limit
            raise ValidationError("Circuit has too many gates")

        # Validate each gate
        for i, gate in enumerate(circuit.gates):
            # Check gate name
            if not gate.name:
                raise ValidationError(f"Gate {i} has no name")

            # Check qubit indices
            for qubit in gate.qubits:
                if qubit < 0 or qubit >= circuit.num_qubits:
                    raise ValidationError(
                        f"Gate {i} ({gate.name}) has invalid qubit index {qubit}"
                    )

            # Validate parameters
            for j, param in enumerate(gate.params):
                if isinstance(param, (int, float)):
                    import math
                    if math.isnan(param):
                        raise ValidationError(
                            f"Gate {i} ({gate.name}) parameter {j} is NaN"
                        )
                    if math.isinf(param):
                        raise ValidationError(
                            f"Gate {i} ({gate.name}) parameter {j} is infinite"
                        )
                    # Check angle bounds (reasonable range)
                    if abs(param) > 1000:
                        raise ValidationError(
                            f"Gate {i} ({gate.name}) parameter {j} "
                            f"value {param} exceeds reasonable bounds"
                        )

    def _marshal_gates(self, circuit: 'Circuit') -> ctypes.Array:
        """
        Convert circuit gates to C structures with validation.

        Args:
            circuit: Circuit containing gates

        Returns:
            Array of MQGate structures
        """
        c_gates = (MQGate * len(circuit.gates))()

        for i, gate in enumerate(circuit.gates):
            gt = c_gates[i]
            gname = gate.name.lower()

            # Map gate name to type ID
            if gname in GATE_MAP:
                gt.type = GATE_MAP[gname]
                gt.num_params = len(gate.params)

                # Validate and copy parameters
                for j, p in enumerate(gate.params):
                    if j < 3:  # C struct has fixed array of 3
                        try:
                            gt.params[j] = float(p)
                        except (TypeError, ValueError):
                            raise InvalidParameterError(
                                f"Gate {i} parameter {j} cannot be converted to float"
                            )

            # Handle special gate mappings
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
                # Unknown gate - log and skip
                logger.warning(f"Unsupported gate '{gname}' at position {i}")
                gt.type = 999  # Invalid type marker

            # Set qubit indices with validation
            gt.num_qubits = len(gate.qubits)
            for j, q in enumerate(gate.qubits):
                if j < 3:  # C struct has fixed array of 3
                    gt.qubits[j] = q

        return c_gates

    def run(
        self,
        circuit: 'Circuit',
        shots: int = 0,
        params: Optional[Union[Dict, List[float]]] = None
    ) -> Dict:
        """
        Execute circuit on GPU with comprehensive error handling.

        Args:
            circuit: Quantum circuit to execute
            shots: Number of measurement shots (0 for statevector)
            params: Optional parameter values

        Returns:
            Dict containing results

        Raises:
            Various exceptions for different error conditions
        """
        start_time = time.perf_counter()

        # Validate inputs
        if shots < 0:
            raise ValidationError("Shots must be non-negative")
        if shots > 1000000:  # Reasonable limit
            raise ValidationError("Shots exceeds maximum (1000000)")

        # Bind parameters if provided
        if params is not None:
            try:
                circuit = circuit.bind_parameters(params)
            except Exception as e:
                raise InvalidParameterError(f"Failed to bind parameters: {e}")

        # Validate circuit
        self._validate_circuit(circuit)

        num_qubits = circuit.num_qubits

        # Marshal gates to C structures
        try:
            c_gates = self._marshal_gates(circuit)
        except Exception as e:
            raise NativeCallError('gate_marshaling', -1, f"Failed to marshal gates: {e}")

        # Allocate output buffers
        sv_size = 1 << num_qubits

        # Check memory requirements
        memory_required = sv_size * 8  # complex64 = 8 bytes
        if memory_required > 16 * 1024**3:  # 16GB limit
            raise ValidationError(
                f"Circuit requires {memory_required / 1024**3:.1f}GB, "
                "exceeding memory limit"
            )

        try:
            statevector = np.zeros(sv_size, dtype=np.complex64)
        except MemoryError:
            raise ValidationError(f"Failed to allocate memory for {num_qubits} qubits")

        sv_ptr = statevector.ctypes.data_as(ctypes.POINTER(MQComplex))

        # Calculate appropriate timeout
        num_gates = len(circuit.gates)
        timeout = self._calculate_timeout(num_qubits, num_gates)

        logger.info(
            f"Executing circuit: {num_qubits} qubits, {num_gates} gates, "
            f"timeout={timeout}s"
        )

        # Execute circuit with timeout
        try:
            result_code = self._wrapper.run_circuit(
                self._ctx,
                num_qubits,
                c_gates,
                num_gates,
                0,  # shots=0 for statevector
                sv_ptr,
                None,  # counts_out
                timeout=timeout
            )
        except Exception as e:
            logger.error(f"Circuit execution failed: {e}")
            raise

        # Build result dictionary
        result = {
            'statevector': statevector,
            'time_ms': (time.perf_counter() - start_time) * 1000
        }

        # Sample if shots requested
        if shots > 0:
            try:
                from ..cpu.measurement import sample_counts
                result['counts'] = sample_counts(statevector, shots, num_qubits)
            except Exception as e:
                logger.error(f"Sampling failed: {e}")
                raise NativeCallError('sampling', -1, f"Failed to sample: {e}")

        return result

    def statevector(self, circuit: 'Circuit', params=None) -> np.ndarray:
        """Get circuit statevector with error handling."""
        res = self.run(circuit, shots=0, params=params)
        return res['statevector']

    def expectation(
        self,
        circuit: 'Circuit',
        hamiltonian: 'Hamiltonian',
        params=None
    ) -> float:
        """Calculate expectation value with error handling."""
        try:
            sv = self.statevector(circuit, params)
            h_matrix = hamiltonian.to_matrix(circuit.num_qubits)
            h_psi = h_matrix @ sv
            return np.real(np.vdot(sv, h_psi))
        except Exception as e:
            logger.error(f"Expectation calculation failed: {e}")
            raise NativeCallError('expectation', -1, str(e))

    def gradient(
        self,
        circuit: 'Circuit',
        hamiltonian: 'Hamiltonian',
        params: List[float],
        method='parameter_shift'
    ) -> np.ndarray:
        """
        Compute gradients with enhanced error handling.

        Args:
            circuit: Parameterized circuit
            hamiltonian: Observable
            params: Parameter values
            method: 'adjoint' or 'parameter_shift'

        Returns:
            Array of gradients
        """
        if method != 'adjoint':
            # Use parameter shift for non-adjoint methods
            from ..cpu.gradient import parameter_shift_gradient
            return parameter_shift_gradient(self, circuit, hamiltonian, params)

        # Validate inputs
        if not params:
            raise ValidationError("No parameters provided for gradient")

        # Bind parameters
        if params is not None:
            circuit = circuit.bind_parameters(params)

        # Validate circuit
        self._validate_circuit(circuit)

        # Marshal gates
        c_gates = self._marshal_gates(circuit)

        # Marshal Hamiltonian
        hamiltonian_struct = self._marshal_hamiltonian(hamiltonian, circuit.num_qubits)

        # Allocate gradient buffer
        total_param_count = sum(len(g.params) for g in circuit.gates)
        grads_out = (ctypes.c_double * total_param_count)()

        # Calculate timeout
        timeout = self._calculate_timeout(circuit.num_qubits, len(circuit.gates)) * 2

        logger.info(f"Computing gradient with timeout={timeout}s")

        # Compute gradients
        try:
            result = self._wrapper.compute_gradient(
                self._ctx,
                circuit.num_qubits,
                c_gates,
                len(circuit.gates),
                hamiltonian_struct,
                grads_out,
                timeout=timeout
            )
        except Exception as e:
            logger.error(f"Gradient computation failed: {e}")
            raise

        # Extract and return gradients
        return self._extract_gradients(circuit, grads_out)

    def _marshal_hamiltonian(
        self,
        hamiltonian: 'Hamiltonian',
        num_qubits: int
    ) -> ctypes.POINTER:
        """Marshal Hamiltonian to C structure."""
        from ...spin import PauliTerm, Hamiltonian

        if isinstance(hamiltonian, PauliTerm):
            hamiltonian = Hamiltonian([hamiltonian])

        terms = hamiltonian.terms
        num_terms = len(terms)

        # Allocate arrays
        coeffs_arr = (ctypes.c_double * num_terms)()
        pauli_codes_arr = (ctypes.c_uint8 * (num_terms * num_qubits))()

        # Initialize
        ctypes.memset(pauli_codes_arr, 0, ctypes.sizeof(pauli_codes_arr))

        # Fill arrays
        for i, term in enumerate(terms):
            coeffs_arr[i] = term.coeff.real

            for p_str, q_idx in term.ops:
                code = 0
                if p_str == 'X': code = 1
                elif p_str == 'Y': code = 2
                elif p_str == 'Z': code = 3

                if q_idx < num_qubits:
                    pauli_codes_arr[i * num_qubits + q_idx] = code

        # Create structure
        h_struct = MQHamiltonian()
        h_struct.num_terms = num_terms
        h_struct.num_qubits = num_qubits
        h_struct.coeffs = ctypes.cast(coeffs_arr, ctypes.POINTER(ctypes.c_double))
        h_struct.pauli_codes = ctypes.cast(pauli_codes_arr, ctypes.POINTER(ctypes.c_uint8))

        return ctypes.byref(h_struct)

    def _extract_gradients(
        self,
        circuit: 'Circuit',
        grads_buffer: ctypes.Array
    ) -> np.ndarray:
        """Extract gradients from buffer."""
        final_grads = []
        grad_ptr = 0

        for gate in circuit.gates:
            num_params = len(gate.params)
            for k in range(num_params):
                final_grads.append(grads_buffer[grad_ptr + k])
            grad_ptr += num_params

        return np.array(final_grads)

    def _calculate_timeout(self, num_qubits: int, num_gates: int) -> float:
        """
        Calculate appropriate timeout based on circuit complexity.

        Args:
            num_qubits: Number of qubits
            num_gates: Number of gates

        Returns:
            Timeout in seconds
        """
        # Base timeout
        base = 10.0

        # Scale with circuit complexity
        # Roughly O(2^n * g) for statevector simulation
        complexity_factor = (2 ** (num_qubits / 10)) * (num_gates / 100)

        # Apply scale factor
        timeout = base + complexity_factor * self.timeout_scale

        # Cap at reasonable maximum (5 minutes by default)
        max_timeout = 300.0 * self.timeout_scale

        return min(timeout, max_timeout)