"""
Tests for Security and Error Handling Enhancements

Tests the comprehensive error handling, timeout mechanisms,
and input validation added to Metal-Q.
"""
import pytest
import numpy as np
import math
import time
import threading
import ctypes
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import the modules to test
from metalq.exceptions import (
    MetalQException,
    NativeLibraryError,
    LibraryLoadError,
    NativeCallError,
    NativeTimeoutError,
    InvalidParameterError,
    ValidationError,
    validate_parameters,
    with_timeout,
    TimeoutHandler,
    ErrorCodeMapper
)
from metalq.circuit import Circuit
from metalq.parameter import Parameter
from metalq.spin import Z


class TestExceptionSanitization:
    """Test error message sanitization."""

    def test_memory_address_sanitization(self):
        """Test that memory addresses are redacted."""
        msg = "Error at memory location 0x7fff8b3c4000"
        exc = MetalQException(msg)
        assert "0x[REDACTED]" in exc.safe_message
        assert "0x7fff8b3c4000" not in exc.safe_message

    def test_file_path_sanitization(self):
        """Test that file paths are sanitized."""
        msg = "Failed to load /Users/testuser/projects/quantum/lib.so"
        exc = MetalQException(msg)
        assert "/Users/[USER]" in exc.safe_message
        assert "testuser" not in exc.safe_message

    def test_windows_path_sanitization(self):
        """Test Windows paths are sanitized."""
        msg = "Error loading C:\\Users\\Admin\\AppData\\lib.dll"
        exc = MetalQException(msg)
        assert "[PATH]" in exc.safe_message
        assert "Admin" not in exc.safe_message

    def test_message_truncation(self):
        """Test that long messages are truncated."""
        msg = "A" * 1000
        exc = MetalQException(msg)
        assert len(exc.safe_message) <= 520  # 500 + truncation message

    def test_nested_exception_details(self):
        """Test that exception details are preserved."""
        exc = NativeCallError("test_func", 42, "Test error")
        assert exc.details["function"] == "test_func"
        assert exc.details["error_code"] == 42


class TestParameterValidation:
    """Test parameter validation functionality."""

    def test_nan_detection(self):
        """Test NaN values are rejected."""
        params = {"angle": float('nan')}
        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(params)
        assert "NaN" in str(exc_info.value)

    def test_infinity_detection(self):
        """Test infinite values are rejected."""
        params = {"angle": float('inf')}
        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(params)
        assert "infinite" in str(exc_info.value)

    def test_angle_bounds(self):
        """Test angle parameters have reasonable bounds."""
        params = {"theta": 1000 * math.pi}
        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(params)
        assert "exceeds reasonable bounds" in str(exc_info.value)

    def test_negative_qubit_index(self):
        """Test negative qubit indices are rejected."""
        params = {"qubit": -1}
        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(params)
        assert "cannot be negative" in str(exc_info.value)

    def test_large_qubit_index(self):
        """Test excessively large qubit indices are rejected."""
        params = {"target_qubit": 200}
        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(params)
        assert "exceeds maximum" in str(exc_info.value)

    def test_string_injection_prevention(self):
        """Test that injection attempts in strings are blocked."""
        injection_attempts = [
            {"path": "../../../etc/passwd"},
            {"name": "<script>alert('xss')</script>"},
            {"query": "DROP TABLE users"},
            {"cmd": "exec('malicious')"},
            {"expr": "eval('code')"}
        ]

        for params in injection_attempts:
            with pytest.raises(ValidationError) as exc_info:
                validate_parameters(params)
            assert "suspicious content" in str(exc_info.value)

    def test_array_size_limit(self):
        """Test that large arrays are rejected."""
        params = {"data": list(range(20000))}
        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(params)
        assert "too large" in str(exc_info.value)

    def test_valid_parameters_pass(self):
        """Test that valid parameters pass validation."""
        params = {
            "theta": math.pi / 2,
            "qubit": 5,
            "name": "test_gate",
            "values": [1.0, 2.0, 3.0]
        }
        validated = validate_parameters(params)
        assert validated == params


class TestTimeoutMechanisms:
    """Test timeout functionality."""

    def test_timeout_handler(self):
        """Test TimeoutHandler context manager."""
        with pytest.raises(NativeTimeoutError):
            with TimeoutHandler(0.1, "test_operation"):
                time.sleep(0.5)

    def test_timeout_handler_no_timeout(self):
        """Test TimeoutHandler when operation completes."""
        with TimeoutHandler(0.5, "test_operation") as handler:
            time.sleep(0.1)
            # Should not raise

    def test_timeout_decorator(self):
        """Test with_timeout decorator."""
        @with_timeout(0.2)
        def slow_function():
            time.sleep(0.5)
            return "done"

        with pytest.raises(NativeTimeoutError):
            slow_function()

    def test_timeout_decorator_success(self):
        """Test with_timeout decorator when function completes."""
        @with_timeout(0.5)
        def fast_function():
            time.sleep(0.1)
            return "done"

        result = fast_function()
        assert result == "done"

    def test_timeout_decorator_with_exception(self):
        """Test with_timeout decorator propagates exceptions."""
        @with_timeout(0.5)
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError) as exc_info:
            failing_function()
        assert "Test error" in str(exc_info.value)


class TestErrorCodeMapping:
    """Test error code to exception mapping."""

    def test_memory_error_codes(self):
        """Test memory error code mapping."""
        exc = ErrorCodeMapper.get_exception(1000, "alloc")
        assert isinstance(exc, MemoryError)
        assert "Failed to allocate memory" in str(exc)

    def test_gpu_error_codes(self):
        """Test GPU error code mapping."""
        from metalq.exceptions import GPUError
        exc = ErrorCodeMapper.get_exception(2000, "init")
        assert isinstance(exc, GPUError)
        assert "GPU device not available" in str(exc)

    def test_parameter_error_codes(self):
        """Test parameter error code mapping."""
        exc = ErrorCodeMapper.get_exception(3000, "validate")
        assert isinstance(exc, InvalidParameterError)
        assert "Invalid number of qubits" in str(exc)

    def test_unknown_error_code(self):
        """Test handling of unknown error codes."""
        exc = ErrorCodeMapper.get_exception(99999, "unknown")
        assert isinstance(exc, NativeCallError)
        assert exc.details["error_code"] == 99999

    def test_success_code(self):
        """Test that success code returns None."""
        exc = ErrorCodeMapper.get_exception(0, "test")
        assert exc is None


class TestNativeWrapper:
    """Test the native wrapper functionality."""

    @patch('metalq.backends.mps.native_wrapper.ctypes.CDLL')
    def test_library_validation(self, mock_cdll):
        """Test library validation checks."""
        from metalq.backends.mps.native_wrapper import SecureLibraryLoader

        # Create temp file with wrong permissions
        with tempfile.NamedTemporaryFile(suffix='.dylib', delete=False) as f:
            temp_path = f.name

        try:
            # Make file world-writable
            os.chmod(temp_path, 0o666)

            loader = SecureLibraryLoader('test.dylib')
            # Should reject world-writable file
            assert not loader._validate_library(temp_path)

        finally:
            os.unlink(temp_path)

    @patch('metalq.backends.mps.native_wrapper.ctypes.CDLL')
    def test_timeout_on_native_call(self, mock_cdll):
        """Test timeout handling in native calls."""
        from metalq.backends.mps.native_wrapper import NativeCallWrapper

        # Create mock library
        mock_lib = MagicMock()
        mock_lib.slow_function = Mock(side_effect=lambda: time.sleep(2))

        wrapper = NativeCallWrapper(mock_lib, default_timeout=0.5)

        with pytest.raises(NativeTimeoutError) as exc_info:
            wrapper.call_with_timeout('slow_function', timeout=0.1)

        assert "slow_function" in str(exc_info.value)
        assert "0.1" in str(exc_info.value)


class TestSecureMPSBackend:
    """Test the secure MPS backend."""

    def test_circuit_validation_empty(self):
        """Test validation rejects empty circuits."""
        from metalq.backends.mps.secure_backend import SecureMPSBackend

        # Mock the wrapper initialization
        with patch('metalq.backends.mps.secure_backend.create_secure_native_wrapper'):
            backend = SecureMPSBackend()
            backend._wrapper = MagicMock()
            backend._ctx = MagicMock()

            # Circuit(0) is rejected at construction, so simulate a corrupted
            # circuit object to exercise the backend-level guard
            circuit = Circuit(1)
            circuit.num_qubits = 0
            with pytest.raises(ValidationError) as exc_info:
                backend._validate_circuit(circuit)
            assert "at least 1 qubit" in str(exc_info.value)

    def test_circuit_validation_too_large(self):
        """Test validation rejects circuits that are too large."""
        from metalq.backends.mps.secure_backend import SecureMPSBackend

        with patch('metalq.backends.mps.secure_backend.create_secure_native_wrapper'):
            backend = SecureMPSBackend()
            backend._wrapper = MagicMock()
            backend._ctx = MagicMock()

            circuit = Circuit(35)  # Exceeds max qubits (30)
            with pytest.raises(ValidationError) as exc_info:
                backend._validate_circuit(circuit)
            assert "maximum is 30" in str(exc_info.value)

    def test_invalid_gate_parameters(self):
        """Test that invalid gate parameters are caught."""
        from metalq.backends.mps.secure_backend import SecureMPSBackend

        with patch('metalq.backends.mps.secure_backend.create_secure_native_wrapper'):
            backend = SecureMPSBackend()
            backend._wrapper = MagicMock()
            backend._ctx = MagicMock()

            circuit = Circuit(2)
            circuit.rx(float('nan'), 0)  # Invalid: NaN parameter

            with pytest.raises(ValidationError) as exc_info:
                backend._validate_circuit(circuit)
            assert "NaN" in str(exc_info.value)

    def test_invalid_qubit_index(self):
        """Test that invalid qubit indices are caught."""
        from metalq.backends.mps.secure_backend import SecureMPSBackend

        with patch('metalq.backends.mps.secure_backend.create_secure_native_wrapper'):
            backend = SecureMPSBackend()
            backend._wrapper = MagicMock()
            backend._ctx = MagicMock()

            circuit = Circuit(2)
            # Manually add invalid gate (bypass circuit validation)
            from metalq.circuit import Gate
            circuit._gates.append(Gate('x', [5], []))  # Invalid index

            with pytest.raises(ValidationError) as exc_info:
                backend._validate_circuit(circuit)
            assert "invalid qubit index" in str(exc_info.value)

    def test_timeout_calculation(self):
        """Test that timeout is calculated based on circuit complexity."""
        from metalq.backends.mps.secure_backend import SecureMPSBackend

        with patch('metalq.backends.mps.secure_backend.create_secure_native_wrapper'):
            backend = SecureMPSBackend(timeout_scale=2.0)

            # Small circuit
            timeout = backend._calculate_timeout(5, 10)
            assert timeout > 10  # Base timeout
            assert timeout < 30  # Reasonable for small circuit

            # Large circuit
            timeout = backend._calculate_timeout(20, 1000)
            assert timeout > 30  # Should be larger
            assert timeout <= 600  # Should respect maximum


class TestMPSBackendTimeoutSafety:
    """Test timeout safety behavior in the legacy MPS backend."""

    def test_context_is_poisoned_after_timed_out_native_call(self):
        """Timed-out context-backed calls must not leave the backend reusable."""
        from metalq.backends.mps.backend import MPSBackend

        mock_lib = MagicMock()
        mock_lib.metalq_create_context = MagicMock(return_value=ctypes.c_void_p(123))
        mock_lib.metalq_destroy_context = MagicMock(return_value=0)
        mock_lib.metalq_is_supported = MagicMock(return_value=True)
        mock_lib.metalq_gradient_adjoint = MagicMock(return_value=0)
        mock_lib.metalq_run = MagicMock(side_effect=lambda *args, **kwargs: time.sleep(0.2))

        with patch.object(MPSBackend, '_load_library', return_value=mock_lib):
            backend = MPSBackend()

        with pytest.raises(NativeTimeoutError):
            backend._call_with_timeout('metalq_run', backend._ctx, timeout=0.01)

        assert backend._ctx is None
        assert backend._ctx_poisoned is True

        with pytest.raises(NativeCallError) as exc_info:
            backend._call_with_timeout('metalq_run', ctypes.c_void_p(123), timeout=0.01)

        assert "timed out" in str(exc_info.value)

        backend.__del__()
        mock_lib.metalq_destroy_context.assert_not_called()


class TestIntegration:
    """Integration tests for the complete error handling system."""

    @pytest.mark.skipif(
        not os.path.exists('/Users/masakishiraishi/projects/Metal-Q/native/build/libmetalq.dylib'),
        reason="Native library not available"
    )
    def test_secure_backend_integration(self):
        """Test secure backend with actual circuit execution."""
        from metalq.backends.mps.secure_backend import SecureMPSBackend

        try:
            backend = SecureMPSBackend()

            # Create simple circuit
            circuit = Circuit(2)
            circuit.h(0)
            circuit.cx(0, 1)

            # Should execute successfully
            result = backend.run(circuit, shots=100)
            assert 'counts' in result or 'statevector' in result
            assert 'time_ms' in result

        except LibraryLoadError:
            pytest.skip("Native library not available")

    def test_error_logging(self):
        """Test that errors are properly logged."""
        import logging
        from metalq.exceptions import MetalQException

        # Capture log output
        with patch.object(logging.getLogger('metalq.exceptions'), 'error') as mock_log:
            exc = MetalQException("Test error", {"key": "value"})

            # Verify logging occurred
            mock_log.assert_called_once()
            log_message = mock_log.call_args[0][0]
            assert "Test error" in log_message


# Performance tests
class TestPerformance:
    """Test performance characteristics of error handling."""

    def test_validation_performance(self):
        """Test that parameter validation is fast enough."""
        import timeit

        params = {
            "theta": math.pi,
            "phi": math.pi/2,
            "qubits": [1, 2, 3],
            "name": "test"
        }

        # Should complete in reasonable time
        time_taken = timeit.timeit(
            lambda: validate_parameters(params),
            number=1000
        )

        assert time_taken < 1.0  # 1000 validations in under 1 second

    def test_exception_creation_performance(self):
        """Test that exception creation is fast."""
        import timeit

        def create_exception():
            return MetalQException(
                "Error at 0x1234567890 in /home/user/file.txt",
                {"code": 42}
            )

        time_taken = timeit.timeit(create_exception, number=1000)
        assert time_taken < 1.0  # 1000 exceptions in under 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
