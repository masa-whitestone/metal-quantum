#!/usr/bin/env python3
"""
Tests for Input Validation and Library Security

Tests the comprehensive input validation and secure library loading
to ensure all security features work correctly.
"""

import pytest
import math
import numpy as np
import os
import tempfile
import stat
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from metalq.validation import (
    NumericalValidator,
    IndexValidator,
    ParameterExpressionValidator,
    CircuitValidator,
    ValidationError,
    NumericalBounds
)
from metalq.secure_loader import (
    SecureLibraryLoader,
    SecureLibraryConfig,
    LibrarySignature,
    LibraryChecksumValidator,
    LibrarySignatureValidator,
    LibrarySecurityError,
    generate_library_signatures,
    save_signatures_to_file,
    load_signatures_from_file
)


class TestNumericalValidator:
    """Test numerical parameter validation."""

    def setup_method(self):
        """Setup for each test."""
        self.validator = NumericalValidator()

    def test_validate_real_valid(self):
        """Test validation of valid real numbers."""
        assert self.validator.validate_real(5.0) == 5.0
        assert self.validator.validate_real(0) == 0.0
        assert self.validator.validate_real(-3.14) == -3.14
        assert self.validator.validate_real(1e-10) == 1e-10

    def test_validate_real_nan(self):
        """Test that NaN is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_real(float('nan'))
        assert "cannot be NaN" in str(exc_info.value)

    def test_validate_real_infinity(self):
        """Test that Infinity is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_real(float('inf'))
        assert "cannot be Infinity" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_real(float('-inf'))
        assert "cannot be Infinity" in str(exc_info.value)

    def test_validate_real_bounds(self):
        """Test bounds checking."""
        # Within bounds
        assert self.validator.validate_real(100, min_val=0, max_val=1000) == 100

        # Outside bounds
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_real(1001, min_val=0, max_val=1000)
        assert "outside acceptable range" in str(exc_info.value)

    def test_validate_angle(self):
        """Test angle validation."""
        assert self.validator.validate_angle(math.pi) == math.pi
        assert self.validator.validate_angle(0) == 0
        assert self.validator.validate_angle(-math.pi/2) == -math.pi/2

        # Large angles should trigger warning but still validate
        large_angle = 10 * math.pi
        assert self.validator.validate_angle(large_angle) == large_angle

    def test_validate_probability(self):
        """Test probability validation."""
        assert self.validator.validate_probability(0.5) == 0.5
        assert self.validator.validate_probability(0) == 0
        assert self.validator.validate_probability(1) == 1

        # Outside [0,1]
        with pytest.raises(ValidationError):
            self.validator.validate_probability(-0.1)

        with pytest.raises(ValidationError):
            self.validator.validate_probability(1.1)

        # Edge cases with floating point
        assert self.validator.validate_probability(-1e-16) == 0.0
        assert self.validator.validate_probability(1 + 1e-16) == 1.0

    def test_validate_complex(self):
        """Test complex number validation."""
        assert self.validator.validate_complex(1+2j) == 1+2j
        assert self.validator.validate_complex(5) == 5+0j
        assert self.validator.validate_complex(3.14j) == 3.14j

        # NaN in complex
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_complex(float('nan') + 1j)
        assert "cannot be NaN" in str(exc_info.value)

        # Infinity in complex
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_complex(1 + float('inf')*1j)
        # Note: inf in imaginary part makes real part nan, so it detects as NaN first
        assert "cannot be" in str(exc_info.value)

        # Pure infinity
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_complex(float('inf'))
        assert "cannot be Infinity" in str(exc_info.value)

    def test_validate_unitary_matrix(self):
        """Test unitary matrix validation."""
        # Valid unitary matrices
        I = np.eye(2)
        assert np.allclose(self.validator.validate_unitary_matrix(I), I)

        X = np.array([[0, 1], [1, 0]])
        assert np.allclose(self.validator.validate_unitary_matrix(X), X)

        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        assert np.allclose(self.validator.validate_unitary_matrix(H), H)

        # Non-unitary matrix
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_unitary_matrix([[1, 0], [0, 2]])
        assert "not unitary" in str(exc_info.value)

        # Non-square matrix
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_unitary_matrix([[1, 0, 0], [0, 1, 0]])
        assert "must be square" in str(exc_info.value)

        # Matrix with NaN
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_unitary_matrix([[1, float('nan')], [0, 1]])
        assert "contains NaN" in str(exc_info.value)


class TestIndexValidator:
    """Test index validation."""

    def test_validate_qubit_index(self):
        """Test single qubit index validation."""
        assert IndexValidator.validate_qubit_index(0, 5) == 0
        assert IndexValidator.validate_qubit_index(4, 5) == 4

        # Out of range
        with pytest.raises(ValidationError) as exc_info:
            IndexValidator.validate_qubit_index(5, 5)
        assert "out of range" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            IndexValidator.validate_qubit_index(-1, 5)
        assert "out of range" in str(exc_info.value)

        # Non-integer
        with pytest.raises(ValidationError) as exc_info:
            IndexValidator.validate_qubit_index(2.5, 5)
        assert "must be an integer" in str(exc_info.value)

    def test_validate_qubit_indices(self):
        """Test multiple qubit indices validation."""
        assert IndexValidator.validate_qubit_indices([0, 1, 2], 5) == [0, 1, 2]
        assert IndexValidator.validate_qubit_indices(3, 5) == [3]

        # Duplicates
        with pytest.raises(ValidationError) as exc_info:
            IndexValidator.validate_qubit_indices([1, 2, 1], 5)
        assert "duplicate indices" in str(exc_info.value)

        # Out of range
        with pytest.raises(ValidationError) as exc_info:
            IndexValidator.validate_qubit_indices([0, 5], 5)
        assert "out of range" in str(exc_info.value)

    def test_validate_bit_index(self):
        """Test classical bit index validation."""
        assert IndexValidator.validate_bit_index(0, 8) == 0
        assert IndexValidator.validate_bit_index(7, 8) == 7

        # Out of range
        with pytest.raises(ValidationError) as exc_info:
            IndexValidator.validate_bit_index(8, 8)
        assert "out of range" in str(exc_info.value)


class TestParameterExpressionValidator:
    """Test parameter expression validation."""

    def test_validate_parameter_name(self):
        """Test parameter name validation."""
        # Valid names
        assert ParameterExpressionValidator.validate_parameter_name("theta") == "theta"
        assert ParameterExpressionValidator.validate_parameter_name("param_1") == "param_1"
        assert ParameterExpressionValidator.validate_parameter_name("_x") == "_x"

        # Invalid names
        with pytest.raises(ValidationError):
            ParameterExpressionValidator.validate_parameter_name("")

        with pytest.raises(ValidationError):
            ParameterExpressionValidator.validate_parameter_name("1param")

        with pytest.raises(ValidationError):
            ParameterExpressionValidator.validate_parameter_name("param-name")

        # Reserved keywords
        with pytest.raises(ValidationError):
            ParameterExpressionValidator.validate_parameter_name("import")

    def test_sanitize_expression(self):
        """Test expression sanitization."""
        # Clean expressions
        assert ParameterExpressionValidator.sanitize_expression("2 * theta + 1") == "2 * theta + 1"

        # Suspicious patterns
        dangerous_expressions = [
            "__import__('os')",
            "eval('code')",
            "exec('code')",
            "open('/etc/passwd')",
            "__builtins__",
            "globals()",
        ]

        for expr in dangerous_expressions:
            with pytest.raises(ValidationError) as exc_info:
                ParameterExpressionValidator.sanitize_expression(expr)
            assert "suspicious pattern" in str(exc_info.value)

        # Non-printable characters (null byte is suspicious)
        expr_with_null = "theta\x00 + 1"
        with pytest.raises(ValidationError) as exc_info:
            ParameterExpressionValidator.sanitize_expression(expr_with_null)
        assert "suspicious pattern" in str(exc_info.value)


class TestCircuitValidator:
    """Test circuit validation."""

    def test_circuit_validator_init(self):
        """Test circuit validator initialization."""
        validator = CircuitValidator(5, 8)
        assert validator.num_qubits == 5
        assert validator.num_bits == 8

        # Invalid qubit count
        with pytest.raises(ValidationError):
            CircuitValidator(0)

        with pytest.raises(ValidationError):
            CircuitValidator(101)

    def test_validate_gate_params(self):
        """Test gate parameter validation."""
        validator = CircuitValidator(5)

        # Rotation gates
        params = validator.validate_gate_params('RX', [math.pi/2])
        assert len(params) == 1
        assert abs(params[0] - math.pi/2) < 1e-10

        # U3 gate
        params = validator.validate_gate_params('U3', [math.pi, math.pi/2, math.pi/4])
        assert len(params) == 3

        # Wrong number of parameters
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_gate_params('RX', [math.pi, math.pi/2])
        assert "expects 1 parameters" in str(exc_info.value)

        # Invalid parameter value
        with pytest.raises(ValidationError):
            validator.validate_gate_params('RX', [float('nan')])

    def test_validate_measurement(self):
        """Test measurement validation."""
        validator = CircuitValidator(5, 8)

        qubit, bit = validator.validate_measurement(2, 3)
        assert qubit == 2
        assert bit == 3

        # Invalid indices
        with pytest.raises(ValidationError):
            validator.validate_measurement(5, 3)

        with pytest.raises(ValidationError):
            validator.validate_measurement(2, 8)


class TestLibraryChecksumValidator:
    """Test library checksum validation."""

    def test_calculate_checksum(self):
        """Test checksum calculation."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content for checksum")
            temp_path = Path(f.name)

        try:
            # Calculate checksums
            sha256 = LibraryChecksumValidator.calculate_checksum(temp_path, 'sha256')
            sha512 = LibraryChecksumValidator.calculate_checksum(temp_path, 'sha512')

            # Verify format
            assert len(sha256) == 64  # SHA-256 hex digest
            assert len(sha512) == 128  # SHA-512 hex digest
            assert all(c in '0123456789abcdef' for c in sha256)

        finally:
            temp_path.unlink()

    def test_verify_checksum(self):
        """Test checksum verification."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = Path(f.name)

        try:
            # Calculate actual checksum
            actual = LibraryChecksumValidator.calculate_checksum(temp_path, 'sha256')

            # Verify correct checksum
            assert LibraryChecksumValidator.verify_checksum(temp_path, actual, 'sha256')

            # Verify incorrect checksum
            assert not LibraryChecksumValidator.verify_checksum(
                temp_path, "0" * 64, 'sha256'
            )

        finally:
            temp_path.unlink()


class TestLibrarySignatureValidator:
    """Test library signature validation."""

    def test_generate_and_verify_signature(self):
        """Test signature generation and verification."""
        key = b"test_key_123"
        validator = LibrarySignatureValidator(key)

        # Create signature
        sig = LibrarySignature(
            path="/path/to/lib.so",
            checksum_sha256="a" * 64,
            checksum_sha512="b" * 128,
            size_bytes=1024,
            platform="Linux",
            version="1.0.0"
        )

        # Generate signature
        sig.signature = validator.generate_signature(sig)
        assert sig.signature is not None
        assert len(sig.signature) == 64  # HMAC-SHA256 hex

        # Verify valid signature
        assert validator.verify_signature(sig)

        # Modify data - signature should fail
        sig.size_bytes = 2048
        assert not validator.verify_signature(sig)


class TestSecureLibraryLoader:
    """Test secure library loading."""

    def test_path_security(self):
        """Test path security checks."""
        config = SecureLibraryConfig(
            require_absolute_path=True,
            restrict_to_bundle=True
        )
        loader = SecureLibraryLoader(config)

        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            allowed_dir = tmpdir_path / "allowed"
            allowed_dir.mkdir()
            lib_path = allowed_dir / "lib.so"
            lib_path.write_text("dummy")

            # Add to allowed paths
            config.allowed_paths = [allowed_dir]

            # Valid path should work
            resolved = loader._resolve_library_path(str(lib_path))
            # On macOS, /var is symlink to /private/var - compare canonical paths
            assert resolved.resolve() == lib_path.resolve()

            # Path outside allowed directory should fail
            outside_path = tmpdir_path / "lib.so"
            outside_path.write_text("dummy")

            with pytest.raises(LibrarySecurityError) as exc_info:
                loader._verify_path_security(outside_path)
            assert "not in allowed directories" in str(exc_info.value)

    def test_file_permission_checks(self):
        """Test file permission validation."""
        config = SecureLibraryConfig(verify_permissions=True)
        loader = SecureLibraryLoader(config)

        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            lib_path = Path(f.name)

        try:
            # Make world-writable (insecure)
            os.chmod(lib_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

            with pytest.raises(LibrarySecurityError) as exc_info:
                loader._verify_file_permissions(lib_path)
            assert "world-writable" in str(exc_info.value)

            # Make secure
            os.chmod(lib_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

            # Should not raise
            loader._verify_file_permissions(lib_path)

        finally:
            lib_path.unlink()

    def test_file_size_checks(self):
        """Test file size validation."""
        config = SecureLibraryConfig(max_library_size=100)
        loader = SecureLibraryLoader(config)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * 200)  # Exceed size limit
            lib_path = Path(f.name)

        try:
            with pytest.raises(LibrarySecurityError) as exc_info:
                loader._verify_file_size(lib_path)
            assert "exceeds size limit" in str(exc_info.value)

        finally:
            lib_path.unlink()

    def test_signature_verification_integration(self):
        """Test full signature verification flow."""
        # Create test library
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            f.write(b"test library content")
            lib_path = Path(f.name)

        try:
            # Generate signature
            signatures = generate_library_signatures([str(lib_path)])
            assert lib_path.name in signatures

            sig = signatures[lib_path.name]

            # Create loader with signature
            config = SecureLibraryConfig(
                require_signature=True,
                require_checksum=True
            )
            config.trusted_signatures = signatures
            loader = SecureLibraryLoader(config)

            # Verification should pass
            loader._verify_library_signature(lib_path, sig)

            # Modify file - verification should fail
            lib_path.write_bytes(b"modified content")

            with pytest.raises(LibrarySecurityError) as exc_info:
                loader._verify_library_signature(lib_path, sig)
            assert "checksum mismatch" in str(exc_info.value).lower()

        finally:
            lib_path.unlink()


class TestSignaturePersistence:
    """Test saving and loading signatures."""

    def test_save_and_load_signatures(self):
        """Test signature file operations."""
        # Create signatures
        sig1 = LibrarySignature(
            path="/path/to/lib1.so",
            checksum_sha256="a" * 64,
            checksum_sha512="b" * 128,
            size_bytes=1024,
            platform="Linux",
            version="1.0.0",
            signature="sig1"
        )

        sig2 = LibrarySignature(
            path="/path/to/lib2.so",
            checksum_sha256="c" * 64,
            checksum_sha512="d" * 128,
            size_bytes=2048,
            platform="Darwin",
            version="2.0.0",
            signature="sig2"
        )

        signatures = {"lib1.so": sig1, "lib2.so": sig2}

        # Save to file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            sig_file = f.name

        try:
            save_signatures_to_file(signatures, sig_file)

            # Verify file permissions (should be owner-only)
            stats = os.stat(sig_file)
            assert stats.st_mode & 0o777 == 0o600

            # Load from file
            loaded = load_signatures_from_file(sig_file)

            assert len(loaded) == 2
            assert "lib1.so" in loaded
            assert "lib2.so" in loaded

            # Verify loaded data
            loaded_sig1 = loaded["lib1.so"]
            assert loaded_sig1.checksum_sha256 == sig1.checksum_sha256
            assert loaded_sig1.size_bytes == sig1.size_bytes
            assert loaded_sig1.platform == sig1.platform

        finally:
            Path(sig_file).unlink()


class TestIntegration:
    """Integration tests for validation and security."""

    def test_secure_backend_initialization(self):
        """Test secure backend initialization."""
        from metalq.backends.mps.secure_mps_backend import SecureMPSBackend, SecureMPSConfig

        config = SecureMPSConfig(
            require_library_signature=False,  # Skip signature for test
            validate_inputs=True
        )

        # Mock the initialization to avoid library loading
        with patch.object(SecureMPSBackend, '_initialize_backend') as mock_init:
            backend = SecureMPSBackend(config)
            mock_init.return_value = None
            backend._initialized = True
            backend.native_lib = MagicMock()

            # Should be initialized
            assert backend.config.validate_inputs
            assert backend.config.require_library_signature == False

    def test_secure_backend_validation(self):
        """Test that backend validates inputs."""
        from metalq.backends.mps.secure_mps_backend import SecureMPSBackend, SecureMPSConfig

        config = SecureMPSConfig(
            require_library_signature=False,
            validate_inputs=True
        )

        # Mock initialization
        with patch.object(SecureMPSBackend, '_initialize_backend'):
            backend = SecureMPSBackend(config)
            backend._initialized = True

            # Mock native library
            mock_lib = MagicMock()
            mock_lib.mps_create_circuit.return_value = 12345  # Mock pointer
            mock_lib.mps_add_gate.return_value = 0
            mock_lib.mps_measure.return_value = 0
            backend.native_lib = mock_lib

            # Test circuit creation validation
            circuit = backend.create_circuit(5)
            assert circuit.n_qubits == 5

            # Invalid qubit count
            with pytest.raises(ValidationError):
                backend.create_circuit(0)

            with pytest.raises(ValidationError):
                backend.create_circuit(100)

            # Test gate parameter validation
            mock_lib.mps_add_gate.return_value = 0

            # Valid gate
            backend.add_gate(circuit, 'RX', [0], [math.pi/2])

            # Invalid qubit index
            with pytest.raises(ValidationError):
                backend.add_gate(circuit, 'RX', [5], [math.pi/2])

            # Invalid parameter (NaN)
            with pytest.raises(ValidationError):
                backend.add_gate(circuit, 'RX', [0], [float('nan')])

            # Test measurement validation
            mock_lib.mps_measure.return_value = 0

            # Valid measurement
            backend.measure(circuit, 2, 2)

            # Invalid qubit
            with pytest.raises(ValidationError):
                backend.measure(circuit, 5, 2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])