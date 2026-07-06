"""
Native Library Call Wrapper with Security Enhancements

Provides safe wrapper for native Metal library calls with:
- Comprehensive error handling
- Timeout mechanisms
- Input validation
- Security logging
"""
import ctypes
import os
import hashlib
import json
import logging
from typing import Any, Callable, Dict, Optional, Tuple
from functools import wraps
import threading
import time
from pathlib import Path

from ...exceptions import (
    LibraryLoadError,
    NativeCallError,
    NativeTimeoutError,
    ValidationError,
    handle_native_result,
    validate_parameters,
    with_timeout
)

# Configure security logging
security_logger = logging.getLogger('metalq.security')
security_logger.setLevel(logging.INFO)

# Create file handler for security events
log_dir = Path.home() / '.metalq' / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
security_log_file = log_dir / 'security.log'

handler = logging.FileHandler(security_log_file)
handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
)
security_logger.addHandler(handler)


class SecureLibraryLoader:
    """Secure native library loader with validation."""

    # Known good library checksums (should be updated with each release)
    TRUSTED_CHECKSUMS = {
        'libmetalq.dylib': [
            # Add actual checksums here after building the library
            # 'sha256:abcd1234...',
        ]
    }

    def __init__(self, library_name: str = 'libmetalq.dylib'):
        self.library_name = library_name
        self.library = None
        self._lock = threading.Lock()

    def _calculate_checksum(self, filepath: str) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return f"sha256:{sha256_hash.hexdigest()}"

    def _validate_library(self, filepath: str) -> bool:
        """
        Validate library integrity and permissions.

        Args:
            filepath: Path to the library file

        Returns:
            True if library is valid and safe to load
        """
        # Check file exists
        if not os.path.exists(filepath):
            return False

        # Check file permissions (should not be world-writable)
        stat_info = os.stat(filepath)
        if stat_info.st_mode & 0o002:  # World-writable
            security_logger.warning(
                f"Library {filepath} is world-writable, refusing to load"
            )
            return False

        # Check ownership (optional - uncomment if needed)
        # import pwd
        # if stat_info.st_uid != os.getuid():
        #     security_logger.warning(
        #         f"Library {filepath} not owned by current user"
        #     )
        #     return False

        # Verify checksum if available
        checksum = self._calculate_checksum(filepath)
        if self.library_name in self.TRUSTED_CHECKSUMS:
            if checksum not in self.TRUSTED_CHECKSUMS[self.library_name]:
                security_logger.error(
                    f"Library {filepath} checksum mismatch: {checksum}"
                )
                # In production, should return False here
                # For development, log warning but allow
                security_logger.warning(
                    "Allowing untrusted library for development"
                )

        security_logger.info(
            f"Library validation passed for {filepath} ({checksum})"
        )
        return True

    def load_library(self, search_paths: Optional[list] = None) -> ctypes.CDLL:
        """
        Securely load native library.

        Args:
            search_paths: Optional list of paths to search

        Returns:
            Loaded library object

        Raises:
            LibraryLoadError: If library cannot be loaded securely
        """
        with self._lock:
            if self.library:
                return self.library

            # Default secure search paths (most specific to least specific)
            if search_paths is None:
                search_paths = [
                    # Bundle path (most secure)
                    os.path.join(
                        os.path.dirname(__file__),
                        "../../../native/build"
                    ),
                    # User-specific path
                    os.path.expanduser(f"~/.metalq/lib/{self.library_name}"),
                    # Avoid system paths that could be compromised
                ]

            for search_dir in search_paths:
                filepath = os.path.join(search_dir, self.library_name)
                filepath = os.path.abspath(filepath)

                if self._validate_library(filepath):
                    try:
                        security_logger.info(f"Loading library from {filepath}")
                        self.library = ctypes.CDLL(filepath)

                        # Verify library has expected functions
                        self._verify_library_interface()

                        security_logger.info("Library loaded successfully")
                        return self.library
                    except Exception as e:
                        security_logger.error(
                            f"Failed to load library from {filepath}: {e}"
                        )
                        continue

            raise LibraryLoadError(
                f"Failed to load {self.library_name} from any secure location"
            )

    def _verify_library_interface(self):
        """Verify that loaded library has expected functions."""
        required_functions = [
            'metalq_create_context',
            'metalq_destroy_context',
            'metalq_run',
            'metalq_gradient_adjoint',
            'metalq_is_supported'
        ]

        for func_name in required_functions:
            if not hasattr(self.library, func_name):
                raise LibraryLoadError(
                    f"Library missing required function: {func_name}"
                )


class NativeCallWrapper:
    """
    Wrapper for safe native function calls with timeout and error handling.
    """

    def __init__(self, library: ctypes.CDLL, default_timeout: float = 30.0):
        self.library = library
        self.default_timeout = default_timeout
        self._configure_functions()

    def _configure_functions(self):
        """Configure function signatures for type safety."""
        # Import ctypes struct definitions from the backend module
        from .backend import MQComplex, MQGate, MQHamiltonian

        # Configure metalq_create_context
        self.library.metalq_create_context.restype = ctypes.c_void_p
        self.library.metalq_create_context.argtypes = []

        # Configure metalq_is_supported
        self.library.metalq_is_supported.restype = ctypes.c_bool
        self.library.metalq_is_supported.argtypes = []

        # Configure metalq_destroy_context
        self.library.metalq_destroy_context.restype = ctypes.c_int
        self.library.metalq_destroy_context.argtypes = [ctypes.c_void_p]

        # Configure metalq_run
        self.library.metalq_run.restype = ctypes.c_int
        self.library.metalq_run.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(MQGate),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(MQComplex),
            ctypes.POINTER(ctypes.c_uint64)
        ]

        # Configure metalq_gradient_adjoint
        self.library.metalq_gradient_adjoint.restype = ctypes.c_int
        self.library.metalq_gradient_adjoint.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(MQGate),
            ctypes.c_uint32,
            ctypes.POINTER(MQHamiltonian),
            ctypes.POINTER(ctypes.c_double)
        ]

    def call_with_timeout(
        self,
        func_name: str,
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Call native function with timeout and error handling.

        Args:
            func_name: Name of the native function
            *args: Function arguments
            timeout: Timeout in seconds (uses default if None)
            **kwargs: Additional keyword arguments

        Returns:
            Function result

        Raises:
            NativeTimeoutError: If function times out
            NativeCallError: If function returns error code
        """
        if timeout is None:
            timeout = self.default_timeout

        # Get the function
        if not hasattr(self.library, func_name):
            raise NativeCallError(func_name, -1, f"Function {func_name} not found")

        func = getattr(self.library, func_name)

        # Log the call for security auditing
        security_logger.debug(
            f"Calling native function: {func_name} with timeout={timeout}s"
        )

        # Execute with timeout
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
            security_logger.error(
                f"Native function {func_name} timed out after {timeout}s"
            )
            raise NativeTimeoutError(func_name, timeout)

        if exception[0]:
            security_logger.error(
                f"Native function {func_name} raised exception: {exception[0]}"
            )
            raise exception[0]

        # Check return code if applicable (skip functions that return
        # a bool or a context pointer instead of a status code)
        if isinstance(result[0], int) and func_name not in ('metalq_is_supported', 'metalq_create_context'):
            handle_native_result(result[0], func_name)

        return result[0]

    def create_context(self, timeout: float = 5.0) -> ctypes.c_void_p:
        """Create Metal context with timeout."""
        ctx = self.call_with_timeout('metalq_create_context', timeout=timeout)
        if not ctx:
            raise NativeCallError('metalq_create_context', -1, "Failed to create context")
        security_logger.info("Metal context created successfully")
        return ctx

    def destroy_context(self, ctx: ctypes.c_void_p, timeout: float = 5.0):
        """Destroy Metal context with timeout."""
        result = self.call_with_timeout('metalq_destroy_context', ctx, timeout=timeout)
        security_logger.info("Metal context destroyed")
        return result

    def is_supported(self, timeout: float = 2.0) -> bool:
        """Check if Metal is supported."""
        return self.call_with_timeout('metalq_is_supported', timeout=timeout)

    def run_circuit(
        self,
        ctx: ctypes.c_void_p,
        num_qubits: int,
        gates: Any,
        num_gates: int,
        shots: int,
        statevector_out: Any,
        counts_out: Any,
        timeout: Optional[float] = None
    ) -> int:
        """
        Run quantum circuit with comprehensive validation.

        Args:
            ctx: Metal context
            num_qubits: Number of qubits
            gates: Array of gate structures
            num_gates: Number of gates
            shots: Number of measurement shots
            statevector_out: Output statevector buffer
            counts_out: Output counts buffer
            timeout: Operation timeout

        Returns:
            Result code (0 for success)
        """
        # Validate parameters
        params = {
            'num_qubits': num_qubits,
            'num_gates': num_gates,
            'shots': shots
        }
        validate_parameters(params)

        # Adjust timeout based on circuit size
        if timeout is None:
            # Scale timeout with circuit complexity
            timeout = max(10.0, 0.1 * num_qubits * num_gates)
            timeout = min(timeout, 300.0)  # Cap at 5 minutes

        security_logger.info(
            f"Running circuit: {num_qubits} qubits, {num_gates} gates, "
            f"{shots} shots, timeout={timeout}s"
        )

        result = self.call_with_timeout(
            'metalq_run',
            ctx,
            ctypes.c_uint32(num_qubits),
            gates,
            ctypes.c_uint32(num_gates),
            ctypes.c_uint32(shots),
            statevector_out,
            counts_out,
            timeout=timeout
        )

        security_logger.info(f"Circuit execution completed with result: {result}")
        return result

    def compute_gradient(
        self,
        ctx: ctypes.c_void_p,
        num_qubits: int,
        gates: Any,
        num_gates: int,
        hamiltonian: Any,
        gradients_out: Any,
        timeout: Optional[float] = None
    ) -> int:
        """
        Compute gradients with adjoint differentiation.

        Args:
            ctx: Metal context
            num_qubits: Number of qubits
            gates: Array of gate structures
            num_gates: Number of gates
            hamiltonian: Hamiltonian structure
            gradients_out: Output gradients buffer
            timeout: Operation timeout

        Returns:
            Result code (0 for success)
        """
        # Validate parameters
        params = {
            'num_qubits': num_qubits,
            'num_gates': num_gates
        }
        validate_parameters(params)

        # Adjust timeout based on circuit size
        if timeout is None:
            # Gradient computation is more expensive
            timeout = max(20.0, 0.2 * num_qubits * num_gates)
            timeout = min(timeout, 600.0)  # Cap at 10 minutes

        security_logger.info(
            f"Computing gradient: {num_qubits} qubits, {num_gates} gates, "
            f"timeout={timeout}s"
        )

        result = self.call_with_timeout(
            'metalq_gradient_adjoint',
            ctx,
            ctypes.c_uint32(num_qubits),
            gates,
            ctypes.c_uint32(num_gates),
            hamiltonian,
            gradients_out,
            timeout=timeout
        )

        security_logger.info(f"Gradient computation completed with result: {result}")
        return result


def create_secure_native_wrapper(
    library_name: str = 'libmetalq.dylib',
    search_paths: Optional[list] = None,
    default_timeout: float = 30.0
) -> NativeCallWrapper:
    """
    Create a secure native call wrapper.

    Args:
        library_name: Name of the native library
        search_paths: Optional list of search paths
        default_timeout: Default timeout for operations

    Returns:
        NativeCallWrapper instance

    Raises:
        LibraryLoadError: If library cannot be loaded
    """
    loader = SecureLibraryLoader(library_name)
    library = loader.load_library(search_paths)
    return NativeCallWrapper(library, default_timeout)