"""
Metal-Q Exception Classes and Error Handling Utilities

Provides comprehensive error handling for native library calls with:
- Custom exception hierarchy for better error categorization
- Error message sanitization to prevent information leakage
- Timeout mechanisms for native operations
"""
from typing import Optional, Any, Dict
import builtins
import re
import logging
from functools import wraps
import signal
import threading
import time

# Configure secure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MetalQException(Exception):
    """Base exception for all Metal-Q errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        # Sanitize the message before storing
        self.safe_message = self._sanitize_message(message)
        self.details = details or {}
        super().__init__(self.safe_message)

        # Log the full error internally (not exposed to user)
        logger.error(f"MetalQException: {message}", extra={"details": details})

    @staticmethod
    def _sanitize_message(message: str) -> str:
        """Remove sensitive information from error messages."""
        # Remove memory addresses
        message = re.sub(r'0x[0-9a-fA-F]+', '0x[REDACTED]', message)

        # Redact home-directory paths first (username + everything below it),
        # so the generic path rule below cannot swallow the /Users//home prefix
        message = re.sub(r'/Users/[\w\-\.]+(?:/[\w\-\.]+)*', '/Users/[USER]', message)
        message = re.sub(r'/home/[\w\-\.]+(?:/[\w\-\.]+)*', '/home/[USER]', message)

        # Remove remaining multi-segment file paths that might reveal system structure
        # ({2,} keeps the already-redacted '/Users' prefix intact)
        message = re.sub(r'(?:/[\w\-\.]+){2,}', '[PATH]', message)
        message = re.sub(r'[A-Z]:\\[\w\-\.\\]+', '[PATH]', message)

        # Limit message length to prevent verbose error dumps
        if len(message) > 500:
            message = message[:500] + "... [truncated]"

        return message


class NativeLibraryError(MetalQException):
    """Raised when native library operations fail."""
    pass


class LibraryLoadError(NativeLibraryError):
    """Raised when the native library cannot be loaded."""
    pass


class NativeCallError(NativeLibraryError):
    """Raised when a native function call fails."""

    def __init__(self, function_name: str, error_code: int, message: str = ""):
        details = {
            "function": function_name,
            "error_code": error_code
        }
        if not message:
            message = f"Native function '{function_name}' failed with code {error_code}"
        super().__init__(message, details)


class NativeTimeoutError(NativeLibraryError):
    """Raised when a native operation times out."""

    def __init__(self, function_name: str, timeout_seconds: float):
        message = f"Native function '{function_name}' timed out after {timeout_seconds} seconds"
        super().__init__(message, {"function": function_name, "timeout": timeout_seconds})


class InvalidParameterError(MetalQException):
    """Raised when invalid parameters are provided."""
    pass


class MemoryError(MetalQException, builtins.MemoryError):
    """Raised when memory allocation fails.

    Also inherits the builtin MemoryError so `except MemoryError` works
    whether callers reference the builtin or this class.
    """
    pass


class GPUError(MetalQException):
    """Raised when GPU operations fail."""
    pass


class ValidationError(MetalQException):
    """Raised when input validation fails."""
    pass


class BackendError(MetalQException):
    """Raised when backend operations fail."""
    pass


class TimeoutHandler:
    """Thread-based timeout handler for native operations."""

    def __init__(self, timeout_seconds: float, error_message: str = "Operation timed out"):
        self.timeout_seconds = timeout_seconds
        self.error_message = error_message
        self._timer = None
        self._timed_out = False
        self._lock = threading.Lock()

    def __enter__(self):
        def timeout_callback():
            with self._lock:
                self._timed_out = True

        self._timer = threading.Timer(self.timeout_seconds, timeout_callback)
        self._timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._timer:
            self._timer.cancel()

        with self._lock:
            if self._timed_out and exc_type is None:
                raise NativeTimeoutError(self.error_message, self.timeout_seconds)

    def check_timeout(self):
        """Check if timeout has occurred during operation."""
        with self._lock:
            if self._timed_out:
                raise NativeTimeoutError(self.error_message, self.timeout_seconds)


def with_timeout(timeout_seconds: float):
    """Decorator to add timeout to functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            function_name = func.__name__

            # Use threading for timeout since signal doesn't work with threads
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)

            if thread.is_alive():
                # Thread is still running, timeout occurred
                logger.warning(f"Function {function_name} timed out after {timeout_seconds}s")
                raise NativeTimeoutError(function_name, timeout_seconds)

            if exception[0]:
                raise exception[0]

            return result[0]

        return wrapper
    return decorator


def validate_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize parameters before passing to native code.

    Args:
        params: Dictionary of parameters to validate

    Returns:
        Validated parameters

    Raises:
        ValidationError: If parameters are invalid
    """
    validated = {}

    for key, value in params.items():
        # Validate numeric parameters
        if isinstance(value, (int, float)):
            # Check for NaN and Infinity
            if not isinstance(value, bool):  # bool is subclass of int
                import math
                if math.isnan(value):
                    raise ValidationError(f"Parameter '{key}' is NaN")
                if math.isinf(value):
                    raise ValidationError(f"Parameter '{key}' is infinite")

                # Check reasonable bounds for quantum parameters
                if key in ['angle', 'theta', 'phi', 'lambda']:
                    if abs(value) > 100 * math.pi:  # Reasonable limit for angles
                        raise ValidationError(
                            f"Parameter '{key}' value {value} exceeds reasonable bounds"
                        )

                # Check qubit indices
                if key in ['qubit', 'target', 'control'] or 'qubit' in key:
                    if value < 0:
                        raise ValidationError(f"Qubit index '{key}' cannot be negative")
                    if value > 100:  # Reasonable maximum
                        raise ValidationError(f"Qubit index '{key}' exceeds maximum")

        # Validate string parameters
        elif isinstance(value, str):
            # Prevent injection attacks
            if len(value) > 1000:
                raise ValidationError(f"String parameter '{key}' too long")

            # Check for suspicious patterns
            suspicious_patterns = [
                r'\.\./',  # Path traversal
                r'<script',  # XSS attempt
                r'DROP\s+TABLE',  # SQL injection
                r'exec\s*\(',  # Code execution
                r'eval\s*\(',  # Code evaluation
            ]

            for pattern in suspicious_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    raise ValidationError(
                        f"Parameter '{key}' contains suspicious content"
                    )

        # Validate arrays
        elif isinstance(value, (list, tuple)):
            if len(value) > 10000:  # Reasonable limit
                raise ValidationError(f"Array parameter '{key}' too large")

            # Recursively validate array elements
            for i, elem in enumerate(value):
                validate_parameters({f"{key}[{i}]": elem})

        validated[key] = value

    return validated


class ErrorCodeMapper:
    """Maps native error codes to appropriate exceptions."""

    # Define error code ranges
    ERROR_CODES = {
        # Native fail-closed codes (small negatives returned by the C layer).
        # -5 = gate encoding failed (unsupported gate reached native code, or a
        # required GPU pipeline was unavailable). Callers normally fail closed in
        # Python before this; it is mapped here for defense in depth.
        -5: (ValidationError, "GPU gate encoding failed: unsupported gate or "
                              "unavailable Metal pipeline"),

        # Memory errors (1000-1999)
        1000: (MemoryError, "Failed to allocate memory"),
        1001: (MemoryError, "Out of GPU memory"),
        1002: (MemoryError, "Memory alignment error"),

        # GPU errors (2000-2999)
        2000: (GPUError, "GPU device not available"),
        2001: (GPUError, "GPU kernel execution failed"),
        2002: (GPUError, "GPU synchronization failed"),

        # Parameter errors (3000-3999)
        3000: (InvalidParameterError, "Invalid number of qubits"),
        3001: (InvalidParameterError, "Invalid gate type"),
        3002: (InvalidParameterError, "Invalid parameter value"),
        3003: (InvalidParameterError, "Parameter out of bounds"),

        # Library errors (4000-4999)
        4000: (NativeLibraryError, "Native library not initialized"),
        4001: (NativeLibraryError, "Function not found in library"),
        4002: (NativeLibraryError, "Library version mismatch"),

        # Generic errors (9000-9999)
        9999: (NativeCallError, "Unknown error occurred"),
    }

    @classmethod
    def get_exception(cls, error_code: int, function_name: str = "unknown") -> Exception:
        """
        Get appropriate exception for error code.

        Args:
            error_code: Native error code
            function_name: Name of the function that failed

        Returns:
            Appropriate exception instance
        """
        if error_code == 0:
            return None  # Success

        # Find matching error code or range
        if error_code in cls.ERROR_CODES:
            exc_class, message = cls.ERROR_CODES[error_code]
            if exc_class == NativeCallError:
                return exc_class(function_name, error_code, message)
            else:
                return exc_class(message, {"error_code": error_code, "function": function_name})

        # Default to NativeCallError for unknown codes
        return NativeCallError(function_name, error_code)


def handle_native_result(result: int, function_name: str = "unknown"):
    """
    Check native function result and raise exception if needed.

    Args:
        result: Return code from native function
        function_name: Name of the native function

    Raises:
        Appropriate exception based on error code
    """
    if result != 0:
        exception = ErrorCodeMapper.get_exception(result, function_name)
        if exception:
            raise exception