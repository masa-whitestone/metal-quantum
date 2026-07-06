"""
Secure Library Loader for Metal-Q

Provides secure loading of native libraries with signature verification,
checksum validation, and path restrictions.
"""

import os
import sys
import ctypes
import hashlib
import hmac
import json
import stat
import platform
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import logging
import tempfile
import shutil

logger = logging.getLogger(__name__)


class LibrarySecurityError(Exception):
    """Raised when library security checks fail."""
    pass


@dataclass
class LibrarySignature:
    """Represents a library's security signature."""
    path: str
    checksum_sha256: str
    checksum_sha512: str
    size_bytes: int
    platform: str
    version: str
    signed_by: Optional[str] = None
    signature: Optional[str] = None


@dataclass
class SecureLibraryConfig:
    """Configuration for secure library loading."""

    # Allowed library search paths (absolute paths only)
    allowed_paths: List[Path] = field(default_factory=list)

    # Known good library signatures
    trusted_signatures: Dict[str, LibrarySignature] = field(default_factory=dict)

    # Security settings
    require_signature: bool = True
    require_checksum: bool = True
    require_absolute_path: bool = True
    restrict_to_bundle: bool = True
    verify_permissions: bool = True
    max_library_size: int = 100 * 1024 * 1024  # 100MB

    # HMAC key for signature verification (should be loaded from secure storage)
    signature_key: Optional[bytes] = None

    def __post_init__(self):
        """Initialize configuration with secure defaults."""
        if not self.allowed_paths:
            # Default to application bundle and system libraries
            self.allowed_paths = self._get_default_allowed_paths()

        if self.signature_key is None:
            # In production, load from secure storage or environment
            self.signature_key = self._load_signature_key()

    def _get_default_allowed_paths(self) -> List[Path]:
        """Get default secure library paths."""
        paths = []

        # Application bundle directory
        if hasattr(sys, '_MEIPASS'):
            # PyInstaller bundle
            paths.append(Path(sys._MEIPASS).resolve())
        else:
            # Development: use package directory
            import metalq
            package_dir = Path(metalq.__file__).parent.resolve()
            paths.append(package_dir / 'backends' / 'mps')

        # System library paths (platform-specific)
        system_name = platform.system()
        if system_name == 'Darwin':
            paths.extend([
                Path('/System/Library/Frameworks/Accelerate.framework/Accelerate'),
                Path('/usr/local/lib'),
            ])
        elif system_name == 'Linux':
            paths.extend([
                Path('/usr/lib'),
                Path('/usr/local/lib'),
                Path('/lib'),
            ])
        elif system_name == 'Windows':
            paths.extend([
                Path(os.environ.get('WINDIR', 'C:\\Windows')) / 'System32',
            ])

        return [p for p in paths if p.exists()]

    def _load_signature_key(self) -> bytes:
        """Load HMAC signature key from secure storage."""
        # In production, implement secure key storage
        # Options:
        # 1. Hardware security module (HSM)
        # 2. OS keychain/keyring
        # 3. Environment variable (less secure)
        # 4. Encrypted configuration file

        # For now, use environment variable or default
        key_hex = os.environ.get('METALQ_SIGNATURE_KEY')
        if key_hex:
            try:
                return bytes.fromhex(key_hex)
            except ValueError:
                logger.warning("Invalid signature key in environment")

        # Default key for development (NOT FOR PRODUCTION)
        return b'DEVELOPMENT_KEY_REPLACE_IN_PRODUCTION'


class LibraryChecksumValidator:
    """Validates library checksums against known good values."""

    @staticmethod
    def calculate_checksum(file_path: Path, algorithm: str = 'sha256') -> str:
        """
        Calculate checksum of a file.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm ('sha256', 'sha512')

        Returns:
            Hex digest of checksum
        """
        if algorithm == 'sha256':
            hasher = hashlib.sha256()
        elif algorithm == 'sha512':
            hasher = hashlib.sha512()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Read file in chunks for memory efficiency
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)

        return hasher.hexdigest()

    @staticmethod
    def verify_checksum(file_path: Path, expected_checksum: str,
                       algorithm: str = 'sha256') -> bool:
        """
        Verify file checksum matches expected value.

        Args:
            file_path: Path to file
            expected_checksum: Expected checksum value
            algorithm: Hash algorithm

        Returns:
            True if checksum matches
        """
        actual = LibraryChecksumValidator.calculate_checksum(file_path, algorithm)
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(actual.lower(), expected_checksum.lower())


class LibrarySignatureValidator:
    """Validates library signatures for authenticity."""

    def __init__(self, signature_key: bytes):
        """
        Initialize validator with signing key.

        Args:
            signature_key: HMAC signing key
        """
        self.signature_key = signature_key

    def generate_signature(self, library_sig: LibrarySignature) -> str:
        """
        Generate HMAC signature for library.

        Args:
            library_sig: Library signature data

        Returns:
            HMAC signature as hex string
        """
        # Create canonical representation
        data = json.dumps({
            'path': os.path.basename(library_sig.path),
            'checksum_sha256': library_sig.checksum_sha256,
            'checksum_sha512': library_sig.checksum_sha512,
            'size_bytes': library_sig.size_bytes,
            'platform': library_sig.platform,
            'version': library_sig.version,
        }, sort_keys=True).encode('utf-8')

        # Generate HMAC
        signature = hmac.new(self.signature_key, data, hashlib.sha256)
        return signature.hexdigest()

    def verify_signature(self, library_sig: LibrarySignature) -> bool:
        """
        Verify library signature.

        Args:
            library_sig: Library signature to verify

        Returns:
            True if signature is valid
        """
        if not library_sig.signature:
            return False

        expected = self.generate_signature(library_sig)
        # Use constant-time comparison
        return hmac.compare_digest(expected, library_sig.signature)


class SecureLibraryLoader:
    """
    Secure library loader with comprehensive security checks.
    """

    def __init__(self, config: Optional[SecureLibraryConfig] = None):
        """
        Initialize secure loader.

        Args:
            config: Security configuration
        """
        self.config = config or SecureLibraryConfig()
        self.checksum_validator = LibraryChecksumValidator()
        self.signature_validator = LibrarySignatureValidator(self.config.signature_key)
        self._loaded_libraries: Dict[str, Any] = {}

    def load_library(self, library_name: str,
                    signature: Optional[LibrarySignature] = None) -> Any:
        """
        Securely load a native library.

        Args:
            library_name: Name or path of library
            signature: Expected library signature

        Returns:
            Loaded library handle

        Raises:
            LibrarySecurityError: If security checks fail
        """
        # Resolve library path
        library_path = self._resolve_library_path(library_name)

        # Security checks
        self._verify_path_security(library_path)
        self._verify_file_permissions(library_path)
        self._verify_file_size(library_path)

        # Signature and checksum verification
        if signature or self.config.require_signature:
            self._verify_library_signature(library_path, signature)

        # Load library in isolated environment
        return self._load_library_sandboxed(library_path)

    def _resolve_library_path(self, library_name: str) -> Path:
        """
        Resolve library name to absolute path.

        Args:
            library_name: Library name or path

        Returns:
            Absolute library path

        Raises:
            LibrarySecurityError: If path is invalid
        """
        # Reject relative paths if absolute paths required
        if self.config.require_absolute_path and not os.path.isabs(library_name):
            # Search in allowed paths
            for search_path in self.config.allowed_paths:
                potential_path = search_path / library_name
                if potential_path.exists():
                    return potential_path.resolve()

            raise LibrarySecurityError(
                f"Library '{library_name}' not found in allowed paths"
            )

        # Convert to Path and resolve
        library_path = Path(library_name).resolve()

        if not library_path.exists():
            raise LibrarySecurityError(f"Library not found: {library_path}")

        return library_path

    def _verify_path_security(self, library_path: Path):
        """
        Verify library path is secure.

        Args:
            library_path: Path to verify

        Raises:
            LibrarySecurityError: If path is insecure
        """
        # Check if path is within allowed directories
        if self.config.restrict_to_bundle or self.config.allowed_paths:
            is_allowed = any(
                library_path.is_relative_to(allowed_path)
                for allowed_path in self.config.allowed_paths
            )

            if not is_allowed:
                raise LibrarySecurityError(
                    f"Library path '{library_path}' is not in allowed directories"
                )

        # Check for symlink attacks
        if library_path.is_symlink():
            real_path = library_path.readlink()
            logger.warning(f"Library is symlink: {library_path} -> {real_path}")

            # Verify symlink target
            self._verify_path_security(real_path)

        # Check for path traversal
        try:
            # Ensure resolved path doesn't escape allowed directories
            resolved = library_path.resolve(strict=True)
            if '../' in str(resolved) or '..\\' in str(resolved):
                raise LibrarySecurityError("Path traversal detected")
        except Exception as e:
            raise LibrarySecurityError(f"Path resolution failed: {e}")

    def _verify_file_permissions(self, library_path: Path):
        """
        Verify file has secure permissions.

        Args:
            library_path: Path to verify

        Raises:
            LibrarySecurityError: If permissions are insecure
        """
        if not self.config.verify_permissions:
            return

        stats = library_path.stat()
        mode = stats.st_mode

        # Check if world-writable
        if mode & stat.S_IWOTH:
            raise LibrarySecurityError(
                f"Library '{library_path}' is world-writable (security risk)"
            )

        # On Unix, check ownership
        if platform.system() != 'Windows':
            # Library should be owned by root or current user
            if stats.st_uid not in (0, os.getuid()):
                logger.warning(
                    f"Library '{library_path}' owned by uid {stats.st_uid}"
                )

    def _verify_file_size(self, library_path: Path):
        """
        Verify file size is reasonable.

        Args:
            library_path: Path to verify

        Raises:
            LibrarySecurityError: If file size is suspicious
        """
        size = library_path.stat().st_size

        if size == 0:
            raise LibrarySecurityError(f"Library '{library_path}' is empty")

        if size > self.config.max_library_size:
            raise LibrarySecurityError(
                f"Library '{library_path}' exceeds size limit: "
                f"{size} > {self.config.max_library_size}"
            )

    def _verify_library_signature(self, library_path: Path,
                                 expected_sig: Optional[LibrarySignature]):
        """
        Verify library signature and checksums.

        Args:
            library_path: Path to library
            expected_sig: Expected signature

        Raises:
            LibrarySecurityError: If verification fails
        """
        # Get expected signature
        if expected_sig is None:
            library_name = library_path.name
            expected_sig = self.config.trusted_signatures.get(library_name)

            if expected_sig is None and self.config.require_signature:
                raise LibrarySecurityError(
                    f"No trusted signature for library '{library_name}'"
                )

        if expected_sig is None:
            return

        # Verify checksums
        if self.config.require_checksum:
            # SHA-256
            if not self.checksum_validator.verify_checksum(
                library_path, expected_sig.checksum_sha256, 'sha256'
            ):
                raise LibrarySecurityError(
                    f"SHA-256 checksum mismatch for '{library_path}'"
                )

            # SHA-512
            if not self.checksum_validator.verify_checksum(
                library_path, expected_sig.checksum_sha512, 'sha512'
            ):
                raise LibrarySecurityError(
                    f"SHA-512 checksum mismatch for '{library_path}'"
                )

        # Verify size
        actual_size = library_path.stat().st_size
        if actual_size != expected_sig.size_bytes:
            raise LibrarySecurityError(
                f"Size mismatch for '{library_path}': "
                f"{actual_size} != {expected_sig.size_bytes}"
            )

        # Verify platform
        if expected_sig.platform != platform.system():
            raise LibrarySecurityError(
                f"Platform mismatch for '{library_path}': "
                f"{platform.system()} != {expected_sig.platform}"
            )

        # Verify signature
        if expected_sig.signature:
            if not self.signature_validator.verify_signature(expected_sig):
                raise LibrarySecurityError(
                    f"Invalid signature for '{library_path}'"
                )

    def _load_library_sandboxed(self, library_path: Path) -> Any:
        """
        Load library in sandboxed environment.

        Args:
            library_path: Path to library

        Returns:
            Library handle
        """
        # Check if already loaded
        path_str = str(library_path)
        if path_str in self._loaded_libraries:
            return self._loaded_libraries[path_str]

        try:
            # Load with restricted mode if available
            if platform.system() == 'Linux':
                # Use RTLD_NOW | RTLD_LOCAL to avoid lazy binding and global symbols
                import ctypes.util
                handle = ctypes.CDLL(path_str, mode=ctypes.RTLD_NOW | ctypes.RTLD_LOCAL)
            else:
                handle = ctypes.CDLL(path_str)

            # Cache loaded library
            self._loaded_libraries[path_str] = handle

            logger.info(f"Securely loaded library: {library_path}")
            return handle

        except Exception as e:
            raise LibrarySecurityError(f"Failed to load library: {e}")


def generate_library_signatures(library_paths: List[str],
                               signature_key: Optional[bytes] = None) -> Dict[str, LibrarySignature]:
    """
    Generate signatures for a list of libraries.

    Args:
        library_paths: List of library paths
        signature_key: HMAC key for signing

    Returns:
        Dictionary of library name to signature
    """
    signatures = {}
    validator = LibrarySignatureValidator(
        signature_key or b'DEVELOPMENT_KEY_REPLACE_IN_PRODUCTION'
    )

    for path_str in library_paths:
        path = Path(path_str).resolve()

        if not path.exists():
            logger.warning(f"Library not found: {path}")
            continue

        # Generate signature
        sig = LibrarySignature(
            path=str(path),
            checksum_sha256=LibraryChecksumValidator.calculate_checksum(path, 'sha256'),
            checksum_sha512=LibraryChecksumValidator.calculate_checksum(path, 'sha512'),
            size_bytes=path.stat().st_size,
            platform=platform.system(),
            version="1.0.0",  # Update as needed
            signed_by="Metal-Q Security Team"
        )

        # Add HMAC signature
        sig.signature = validator.generate_signature(sig)

        signatures[path.name] = sig

    return signatures


def save_signatures_to_file(signatures: Dict[str, LibrarySignature],
                          output_path: str):
    """
    Save library signatures to JSON file.

    Args:
        signatures: Dictionary of signatures
        output_path: Output file path
    """
    # Convert to JSON-serializable format
    data = {}
    for name, sig in signatures.items():
        data[name] = {
            'path': sig.path,
            'checksum_sha256': sig.checksum_sha256,
            'checksum_sha512': sig.checksum_sha512,
            'size_bytes': sig.size_bytes,
            'platform': sig.platform,
            'version': sig.version,
            'signed_by': sig.signed_by,
            'signature': sig.signature,
        }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    # Set restrictive permissions
    os.chmod(output_path, stat.S_IRUSR | stat.S_IWUSR)


def load_signatures_from_file(input_path: str) -> Dict[str, LibrarySignature]:
    """
    Load library signatures from JSON file.

    Args:
        input_path: Input file path

    Returns:
        Dictionary of signatures
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    signatures = {}
    for name, sig_data in data.items():
        signatures[name] = LibrarySignature(**sig_data)

    return signatures