"""
Secure MPS Backend with Input Validation and Library Verification

Integrates comprehensive input validation and secure library loading
for the Metal Performance Shaders quantum simulator.
"""

import os
import sys
import ctypes
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass

# Import security modules
from metalq.validation import (
    NumericalValidator,
    IndexValidator,
    ParameterExpressionValidator,
    CircuitValidator,
    ValidationError
)
from metalq.secure_loader import (
    SecureLibraryLoader,
    SecureLibraryConfig,
    LibrarySignature,
    LibrarySecurityError
)
from metalq.exceptions import (
    MetalQException,
    BackendError,
    InvalidParameterError
)

logger = logging.getLogger(__name__)


@dataclass
class SecureMPSConfig:
    """Configuration for secure MPS backend."""

    # Validation settings
    validate_inputs: bool = True
    strict_bounds: bool = True

    # Library security settings
    require_library_signature: bool = True
    restrict_library_paths: bool = True

    # Performance settings
    timeout_scale: float = 1.0
    max_circuit_depth: int = 10000
    max_qubits: int = 30

    # Paths
    library_signature_file: Optional[Path] = None


class SecureMPSBackend:
    """
    Secure MPS backend with comprehensive validation.

    Features:
    - Input validation for all parameters
    - Secure library loading with signature verification
    - Bounds checking for quantum operations
    - Protection against malicious inputs
    """

    def __init__(self, config: Optional[SecureMPSConfig] = None):
        """
        Initialize secure MPS backend.

        Args:
            config: Security configuration
        """
        self.config = config or SecureMPSConfig()
        self.num_validator = NumericalValidator()
        self.library_loader = None
        self.native_lib = None
        self._initialized = False

        # Initialize backend
        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the backend with security checks."""
        try:
            # Setup secure library configuration
            lib_config = self._create_library_config()
            self.library_loader = SecureLibraryLoader(lib_config)

            # Load native library with verification
            self.native_lib = self._load_native_library()

            # Verify library functions
            self._verify_library_functions()

            self._initialized = True
            logger.info("Secure MPS backend initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize secure backend: {e}")
            raise BackendError(f"Backend initialization failed: {e}")

    def _create_library_config(self) -> SecureLibraryConfig:
        """Create secure library configuration."""
        config = SecureLibraryConfig(
            require_signature=self.config.require_library_signature,
            restrict_to_bundle=self.config.restrict_library_paths,
            verify_permissions=True
        )

        # Load trusted signatures if available
        if self.config.library_signature_file and self.config.library_signature_file.exists():
            from metalq.secure_loader import load_signatures_from_file
            config.trusted_signatures = load_signatures_from_file(
                str(self.config.library_signature_file)
            )

        return config

    def _load_native_library(self):
        """Load native library with security verification."""
        # Determine library name based on platform
        import platform
        system = platform.system()

        if system == 'Darwin':
            lib_name = 'libmetalq.dylib'
        elif system == 'Linux':
            lib_name = 'libmetalq.so'
        elif system == 'Windows':
            lib_name = 'metalq.dll'
        else:
            raise BackendError(f"Unsupported platform: {system}")

        # Try to find library
        search_paths = [
            Path(__file__).parent,  # Same directory as this file
            Path(__file__).parent / 'lib',  # lib subdirectory
        ]

        for search_path in search_paths:
            lib_path = search_path / lib_name
            if lib_path.exists():
                # Load with security checks
                return self.library_loader.load_library(str(lib_path))

        raise BackendError(f"Native library '{lib_name}' not found")

    def _verify_library_functions(self):
        """Verify that required functions exist in library."""
        required_functions = [
            'mps_initialize',
            'mps_create_circuit',
            'mps_add_gate',
            'mps_measure',
            'mps_run',
            'mps_destroy_circuit',
            'mps_cleanup'
        ]

        for func_name in required_functions:
            if not hasattr(self.native_lib, func_name):
                raise BackendError(f"Required function '{func_name}' not found in library")

            # Set function signatures for type safety
            func = getattr(self.native_lib, func_name)
            if func_name == 'mps_create_circuit':
                func.argtypes = [ctypes.c_int, ctypes.c_int]
                func.restype = ctypes.c_void_p
            elif func_name == 'mps_add_gate':
                func.argtypes = [
                    ctypes.c_void_p,  # circuit
                    ctypes.c_char_p,  # gate_name
                    ctypes.POINTER(ctypes.c_int),  # qubits
                    ctypes.c_int,  # num_qubits
                    ctypes.POINTER(ctypes.c_double),  # params
                    ctypes.c_int  # num_params
                ]
                func.restype = ctypes.c_int

    def create_circuit(self, n_qubits: int, n_bits: Optional[int] = None):
        """
        Create a quantum circuit with validation.

        Args:
            n_qubits: Number of qubits
            n_bits: Number of classical bits

        Returns:
            Circuit handle

        Raises:
            ValidationError: If parameters are invalid
        """
        if not self._initialized:
            raise BackendError("Backend not initialized")

        # Validate inputs
        if self.config.validate_inputs:
            if not isinstance(n_qubits, int) or n_qubits <= 0:
                raise ValidationError(f"Invalid number of qubits: {n_qubits}")

            if n_qubits > self.config.max_qubits:
                raise ValidationError(
                    f"Too many qubits: {n_qubits} > {self.config.max_qubits}"
                )

            if n_bits is not None:
                if not isinstance(n_bits, int) or n_bits <= 0:
                    raise ValidationError(f"Invalid number of bits: {n_bits}")

        n_bits = n_bits or n_qubits

        # Create circuit
        circuit_ptr = self.native_lib.mps_create_circuit(n_qubits, n_bits)
        if not circuit_ptr:
            raise BackendError("Failed to create circuit")

        # Return wrapped circuit with validation
        return SecureCircuit(circuit_ptr, n_qubits, n_bits, self)

    def add_gate(self, circuit, gate_name: str, qubits: List[int],
                params: Optional[List[float]] = None):
        """
        Add gate to circuit with validation.

        Args:
            circuit: Circuit handle
            gate_name: Name of quantum gate
            qubits: Qubit indices
            params: Gate parameters

        Raises:
            ValidationError: If inputs are invalid
        """
        if not self._initialized:
            raise BackendError("Backend not initialized")

        # Get circuit info
        if hasattr(circuit, '_ptr'):
            circuit_ptr = circuit._ptr
            n_qubits = circuit.n_qubits
        else:
            raise ValidationError("Invalid circuit object")

        # Validate inputs
        if self.config.validate_inputs:
            # Validate gate name
            gate_name = str(gate_name).upper()
            if not gate_name.replace('_', '').isalnum():
                raise ValidationError(f"Invalid gate name: {gate_name}")

            # Validate qubit indices
            validated_qubits = IndexValidator.validate_qubit_indices(
                qubits, n_qubits, f"gate_{gate_name}_qubits"
            )

            # Validate parameters
            validated_params = []
            if params:
                param_validator = ParameterExpressionValidator()

                for i, param in enumerate(params):
                    # Check for expressions
                    if isinstance(param, str):
                        param = param_validator.sanitize_expression(param)
                        # For now, evaluate simple numeric strings
                        try:
                            param = float(param)
                        except ValueError:
                            raise ValidationError(f"Invalid parameter expression: {param}")

                    # Validate numeric value
                    validated_params.append(
                        self.num_validator.validate_angle(
                            param, f"gate_{gate_name}_param_{i}"
                        )
                    )
        else:
            validated_qubits = list(qubits)
            validated_params = list(params) if params else []

        # Convert to ctypes
        qubits_arr = (ctypes.c_int * len(validated_qubits))(*validated_qubits)
        params_arr = None
        n_params = 0

        if validated_params:
            params_arr = (ctypes.c_double * len(validated_params))(*validated_params)
            n_params = len(validated_params)

        # Add gate
        result = self.native_lib.mps_add_gate(
            circuit_ptr,
            gate_name.encode('utf-8'),
            qubits_arr,
            len(validated_qubits),
            params_arr,
            n_params
        )

        if result != 0:
            raise BackendError(f"Failed to add gate {gate_name}")

    def measure(self, circuit, qubit: int, bit: int):
        """
        Add measurement with validation.

        Args:
            circuit: Circuit handle
            qubit: Qubit to measure
            bit: Classical bit for result

        Raises:
            ValidationError: If inputs are invalid
        """
        if not self._initialized:
            raise BackendError("Backend not initialized")

        # Get circuit info
        if hasattr(circuit, '_ptr'):
            circuit_ptr = circuit._ptr
            n_qubits = circuit.n_qubits
            n_bits = circuit.n_bits
        else:
            raise ValidationError("Invalid circuit object")

        # Validate inputs
        if self.config.validate_inputs:
            qubit = IndexValidator.validate_qubit_index(qubit, n_qubits, "measurement_qubit")
            bit = IndexValidator.validate_bit_index(bit, n_bits, "measurement_bit")

        # Add measurement
        result = self.native_lib.mps_measure(circuit_ptr, qubit, bit)

        if result != 0:
            raise BackendError(f"Failed to add measurement")

    def run(self, circuit, shots: int = 1024) -> Dict[str, int]:
        """
        Run circuit with validation.

        Args:
            circuit: Circuit to run
            shots: Number of shots

        Returns:
            Dictionary of measurement outcomes and counts

        Raises:
            ValidationError: If inputs are invalid
        """
        if not self._initialized:
            raise BackendError("Backend not initialized")

        # Validate shots
        if self.config.validate_inputs:
            if not isinstance(shots, int) or shots <= 0:
                raise ValidationError(f"Invalid number of shots: {shots}")

            if shots > 1000000:
                raise ValidationError(f"Too many shots: {shots} > 1000000")

        # Get circuit pointer
        if hasattr(circuit, '_ptr'):
            circuit_ptr = circuit._ptr
        else:
            raise ValidationError("Invalid circuit object")

        # Run circuit
        results = self.native_lib.mps_run(circuit_ptr, shots)

        if not results:
            raise BackendError("Circuit execution failed")

        # Convert results
        return self._convert_results(results, shots)

    def _convert_results(self, results_ptr, shots: int) -> Dict[str, int]:
        """Convert native results to Python dictionary."""
        # Implementation depends on native library format
        # This is a placeholder
        return {"000": shots//2, "111": shots//2}

    def cleanup(self):
        """Clean up resources."""
        if self.native_lib:
            self.native_lib.mps_cleanup()
        self._initialized = False


class SecureCircuit:
    """
    Secure wrapper for quantum circuit with validation.
    """

    def __init__(self, ptr, n_qubits: int, n_bits: int, backend: SecureMPSBackend):
        """Initialize secure circuit."""
        self._ptr = ptr
        self.n_qubits = n_qubits
        self.n_bits = n_bits
        self.backend = backend
        self.gate_count = 0
        self.depth = 0

    def add_gate(self, gate_name: str, qubits: List[int],
                params: Optional[List[float]] = None):
        """Add gate with validation."""
        self.backend.add_gate(self, gate_name, qubits, params)
        self.gate_count += 1

        # Track depth
        self.depth += 1
        if self.depth > self.backend.config.max_circuit_depth:
            raise ValidationError(
                f"Circuit depth {self.depth} exceeds maximum "
                f"{self.backend.config.max_circuit_depth}"
            )

    def measure(self, qubit: int, bit: int):
        """Add measurement with validation."""
        self.backend.measure(self, qubit, bit)

    def run(self, shots: int = 1024) -> Dict[str, int]:
        """Run circuit with validation."""
        return self.backend.run(self, shots)

    def __del__(self):
        """Clean up circuit resources."""
        if hasattr(self, '_ptr') and self._ptr:
            try:
                self.backend.native_lib.mps_destroy_circuit(self._ptr)
            except:
                pass  # Ignore errors during cleanup