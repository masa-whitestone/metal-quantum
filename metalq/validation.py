"""
Input Validation Module for Metal-Q

Provides comprehensive validation for numerical parameters, qubit indices,
and other inputs to prevent security issues and ensure data integrity.
"""

import math
import cmath
import numpy as np
from typing import Union, List, Tuple, Optional, Any, Dict
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


@dataclass
class NumericalBounds:
    """Defines acceptable bounds for numerical parameters."""
    min_value: float = -1e308  # Near float min
    max_value: float = 1e308   # Near float max
    min_angle: float = -100 * math.pi  # Reasonable angle range
    max_angle: float = 100 * math.pi
    max_abs_complex: float = 1e10  # Maximum absolute value for complex numbers
    epsilon: float = 1e-15  # Minimum non-zero value


class NumericalValidator:
    """
    Validates numerical inputs for quantum operations.

    Checks for:
    - NaN and Infinity values
    - Reasonable bounds
    - Type correctness
    - Special quantum constraints
    """

    def __init__(self, bounds: Optional[NumericalBounds] = None):
        """
        Initialize validator with specified bounds.

        Args:
            bounds: Custom bounds for validation, uses defaults if None
        """
        self.bounds = bounds or NumericalBounds()

    def validate_real(self, value: Any, name: str = "value",
                     min_val: Optional[float] = None,
                     max_val: Optional[float] = None) -> float:
        """
        Validate a real number parameter.

        Args:
            value: Value to validate
            name: Parameter name for error messages
            min_val: Override minimum bound
            max_val: Override maximum bound

        Returns:
            Validated float value

        Raises:
            ValidationError: If validation fails
        """
        # Type check
        if not isinstance(value, (int, float, np.number)):
            raise ValidationError(
                f"{name} must be a real number, got {type(value).__name__}"
            )

        # Convert to float
        try:
            val = float(value)
        except (ValueError, OverflowError) as e:
            raise ValidationError(f"Cannot convert {name} to float: {e}")

        # Check for special values
        if math.isnan(val):
            raise ValidationError(f"{name} cannot be NaN")

        if math.isinf(val):
            raise ValidationError(f"{name} cannot be Infinity")

        # Check bounds
        min_bound = min_val if min_val is not None else self.bounds.min_value
        max_bound = max_val if max_val is not None else self.bounds.max_value

        if val < min_bound or val > max_bound:
            raise ValidationError(
                f"{name} = {val} is outside acceptable range [{min_bound}, {max_bound}]"
            )

        return val

    def validate_angle(self, value: Any, name: str = "angle") -> float:
        """
        Validate an angle parameter (in radians).

        Args:
            value: Angle value to validate
            name: Parameter name for error messages

        Returns:
            Validated angle in radians

        Raises:
            ValidationError: If validation fails
        """
        angle = self.validate_real(
            value, name,
            min_val=self.bounds.min_angle,
            max_val=self.bounds.max_angle
        )

        # Normalize to [-2π, 2π] for efficiency
        # (Multiple rotations are redundant in quantum mechanics)
        if abs(angle) > 2 * math.pi:
            logger.warning(
                f"{name} = {angle} rad is large; consider normalizing to [-2π, 2π]"
            )

        return angle

    def validate_probability(self, value: Any, name: str = "probability") -> float:
        """
        Validate a probability value.

        Args:
            value: Probability to validate
            name: Parameter name for error messages

        Returns:
            Validated probability in [0, 1]

        Raises:
            ValidationError: If validation fails
        """
        # First convert and check special values
        val = self.validate_real(value, name, min_val=-self.bounds.epsilon, max_val=1+self.bounds.epsilon)

        # Handle floating point errors near boundaries
        if val < 0:
            if val > -self.bounds.epsilon:
                return 0.0
            else:
                raise ValidationError(f"{name} = {val} is negative (< 0)")

        if val > 1:
            if val < 1 + self.bounds.epsilon:
                return 1.0
            else:
                raise ValidationError(f"{name} = {val} exceeds 1")

        return val

    def validate_complex(self, value: Any, name: str = "value") -> complex:
        """
        Validate a complex number parameter.

        Args:
            value: Complex value to validate
            name: Parameter name for error messages

        Returns:
            Validated complex number

        Raises:
            ValidationError: If validation fails
        """
        # Type check
        if not isinstance(value, (int, float, complex, np.number)):
            raise ValidationError(
                f"{name} must be a number, got {type(value).__name__}"
            )

        # Convert to complex
        try:
            val = complex(value)
        except (ValueError, OverflowError) as e:
            raise ValidationError(f"Cannot convert {name} to complex: {e}")

        # Check for special values
        # cmath.isnan and cmath.isinf check if ANY part is nan/inf
        if cmath.isnan(val):
            raise ValidationError(f"{name} cannot be NaN")

        if cmath.isinf(val):
            raise ValidationError(f"{name} cannot be Infinity")

        # Check magnitude
        magnitude = abs(val)
        if magnitude > self.bounds.max_abs_complex:
            raise ValidationError(
                f"{name} magnitude {magnitude} exceeds maximum {self.bounds.max_abs_complex}"
            )

        return val

    def validate_unitary_matrix(self, matrix: Any, name: str = "matrix",
                               dim: Optional[int] = None) -> np.ndarray:
        """
        Validate a unitary matrix.

        Args:
            matrix: Matrix to validate
            name: Parameter name for error messages
            dim: Expected dimension (2^n_qubits)

        Returns:
            Validated numpy array

        Raises:
            ValidationError: If validation fails
        """
        # Convert to numpy array
        try:
            mat = np.array(matrix, dtype=complex)
        except Exception as e:
            raise ValidationError(f"Cannot convert {name} to matrix: {e}")

        # Check shape
        if mat.ndim != 2:
            raise ValidationError(f"{name} must be 2-dimensional, got shape {mat.shape}")

        if mat.shape[0] != mat.shape[1]:
            raise ValidationError(f"{name} must be square, got shape {mat.shape}")

        if dim is not None and mat.shape[0] != dim:
            raise ValidationError(
                f"{name} dimension {mat.shape[0]} doesn't match expected {dim}"
            )

        # Check for special values
        if np.any(np.isnan(mat)):
            raise ValidationError(f"{name} contains NaN values")

        if np.any(np.isinf(mat)):
            raise ValidationError(f"{name} contains Infinity values")

        # Check unitarity (U†U = I)
        tolerance = 1e-10
        identity = np.eye(mat.shape[0], dtype=complex)
        product = mat.conj().T @ mat

        if not np.allclose(product, identity, atol=tolerance):
            raise ValidationError(f"{name} is not unitary (U†U ≠ I)")

        return mat


class IndexValidator:
    """
    Validates qubit and bit indices for quantum circuits.
    """

    @staticmethod
    def validate_qubit_index(index: Any, num_qubits: int,
                           name: str = "qubit") -> int:
        """
        Validate a single qubit index.

        Args:
            index: Qubit index to validate
            num_qubits: Total number of qubits in circuit
            name: Parameter name for error messages

        Returns:
            Validated integer index

        Raises:
            ValidationError: If validation fails
        """
        # Type check
        if not isinstance(index, (int, np.integer)):
            raise ValidationError(
                f"{name} index must be an integer, got {type(index).__name__}"
            )

        idx = int(index)

        # Range check
        if idx < 0 or idx >= num_qubits:
            raise ValidationError(
                f"{name} index {idx} out of range [0, {num_qubits-1}]"
            )

        return idx

    @staticmethod
    def validate_qubit_indices(indices: Any, num_qubits: int,
                              name: str = "qubits") -> List[int]:
        """
        Validate a list of qubit indices.

        Args:
            indices: List of qubit indices to validate
            num_qubits: Total number of qubits in circuit
            name: Parameter name for error messages

        Returns:
            Validated list of integer indices

        Raises:
            ValidationError: If validation fails
        """
        # Convert to list
        if isinstance(indices, (int, np.integer)):
            indices = [indices]

        try:
            idx_list = list(indices)
        except TypeError:
            raise ValidationError(f"{name} must be a list or iterable of indices")

        # Validate each index
        validated = []
        for i, idx in enumerate(idx_list):
            validated.append(
                IndexValidator.validate_qubit_index(
                    idx, num_qubits, f"{name}[{i}]"
                )
            )

        # Check for duplicates
        if len(validated) != len(set(validated)):
            raise ValidationError(f"{name} contains duplicate indices")

        return validated

    @staticmethod
    def validate_bit_index(index: Any, num_bits: int,
                          name: str = "bit") -> int:
        """
        Validate a classical bit index.

        Args:
            index: Bit index to validate
            num_bits: Total number of classical bits
            name: Parameter name for error messages

        Returns:
            Validated integer index

        Raises:
            ValidationError: If validation fails
        """
        # Type check
        if not isinstance(index, (int, np.integer)):
            raise ValidationError(
                f"{name} index must be an integer, got {type(index).__name__}"
            )

        idx = int(index)

        # Range check
        if idx < 0 or idx >= num_bits:
            raise ValidationError(
                f"{name} index {idx} out of range [0, {num_bits-1}]"
            )

        return idx


class ParameterExpressionValidator:
    """
    Validates and sanitizes parameter expressions before evaluation.
    """

    # Regex patterns for validation
    VALID_PARAM_NAME = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    SUSPICIOUS_PATTERNS = [
        r'__[a-zA-Z_]+__',  # Dunder methods
        r'import',           # Import statements
        r'exec',             # Exec calls
        r'eval',             # Eval calls
        r'compile',          # Compile calls
        r'globals',          # Global access
        r'locals',           # Local access
        r'vars',             # Variable access
        r'dir',              # Directory listing
        r'open',             # File operations
        r'file',             # File access
        r'input',            # Input operations
        r'raw_input',        # Raw input (Python 2)
        r'__builtins__',     # Builtins access
        r'\x00',             # Null bytes
        r'\\x[0-9a-fA-F]{2}', # Hex escapes
        r'\\[0-7]{1,3}',     # Octal escapes
    ]

    @classmethod
    def validate_parameter_name(cls, name: str) -> str:
        """
        Validate a parameter name.

        Args:
            name: Parameter name to validate

        Returns:
            Validated name

        Raises:
            ValidationError: If name is invalid
        """
        if not isinstance(name, str):
            raise ValidationError(f"Parameter name must be string, got {type(name).__name__}")

        # Check length
        if len(name) == 0:
            raise ValidationError("Parameter name cannot be empty")

        if len(name) > 100:
            raise ValidationError(f"Parameter name too long: {len(name)} > 100")

        # Check format
        if not cls.VALID_PARAM_NAME.match(name):
            raise ValidationError(
                f"Invalid parameter name '{name}'. "
                "Must start with letter/underscore and contain only alphanumeric/underscore"
            )

        # Check for reserved words
        import keyword
        if keyword.iskeyword(name):
            raise ValidationError(f"Parameter name '{name}' is a reserved keyword")

        return name

    @classmethod
    def sanitize_expression(cls, expression: str) -> str:
        """
        Sanitize a parameter expression for safe evaluation.

        Args:
            expression: Expression string to sanitize

        Returns:
            Sanitized expression

        Raises:
            ValidationError: If expression contains suspicious patterns
        """
        if not isinstance(expression, str):
            raise ValidationError(f"Expression must be string, got {type(expression).__name__}")

        # Check length
        if len(expression) > 1000:
            raise ValidationError(f"Expression too long: {len(expression)} > 1000")

        # Check for suspicious patterns
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, expression, re.IGNORECASE):
                raise ValidationError(
                    f"Expression contains suspicious pattern: {pattern}"
                )

        # Remove any non-printable characters
        sanitized = ''.join(c for c in expression if c.isprintable() or c.isspace())

        if sanitized != expression:
            logger.warning("Removed non-printable characters from expression")

        return sanitized

    @classmethod
    def validate_parameter_values(cls, param_values: Dict[Any, Any],
                                 expected_params: Optional[set] = None) -> Dict[str, float]:
        """
        Validate parameter value mappings.

        Args:
            param_values: Dictionary of parameter values
            expected_params: Set of expected parameter names/objects

        Returns:
            Validated dictionary with string keys and float values

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(param_values, dict):
            raise ValidationError(f"Parameter values must be dict, got {type(param_values).__name__}")

        validated = {}
        num_validator = NumericalValidator()

        for key, value in param_values.items():
            # Validate key
            if hasattr(key, 'name'):
                # Parameter object
                key_str = str(key.name)
            else:
                key_str = str(key)

            key_str = cls.validate_parameter_name(key_str)

            # Check if expected
            if expected_params and key_str not in expected_params:
                logger.warning(f"Unexpected parameter '{key_str}' in values")

            # Validate value
            validated[key_str] = num_validator.validate_real(
                value, f"Parameter '{key_str}'"
            )

        return validated


class CircuitValidator:
    """
    Validates quantum circuit parameters and operations.
    """

    def __init__(self, num_qubits: int, num_bits: Optional[int] = None):
        """
        Initialize circuit validator.

        Args:
            num_qubits: Number of qubits in circuit
            num_bits: Number of classical bits (defaults to num_qubits)
        """
        if num_qubits <= 0 or num_qubits > 100:
            raise ValidationError(f"Invalid number of qubits: {num_qubits}")

        self.num_qubits = num_qubits
        self.num_bits = num_bits or num_qubits
        self.num_validator = NumericalValidator()

    def validate_gate_params(self, gate_name: str, params: List[Any]) -> List[float]:
        """
        Validate parameters for a specific gate.

        Args:
            gate_name: Name of the gate
            params: Gate parameters

        Returns:
            Validated parameters

        Raises:
            ValidationError: If validation fails
        """
        gate_name = gate_name.upper()

        # Define expected parameter counts and types
        gate_params = {
            'RX': (1, 'angle'),
            'RY': (1, 'angle'),
            'RZ': (1, 'angle'),
            'PHASE': (1, 'angle'),
            'U': (3, 'angle'),
            'U3': (3, 'angle'),
            'U2': (2, 'angle'),
            'U1': (1, 'angle'),
            'CRZ': (1, 'angle'),
            'CRY': (1, 'angle'),
            'CRX': (1, 'angle'),
            'RSWAP': (1, 'angle'),
            'RZERO': (1, 'probability'),
        }

        if gate_name not in gate_params:
            # Gate doesn't have parameters or is custom
            if params:
                logger.warning(f"Unexpected parameters for gate {gate_name}")
            return []

        expected_count, param_type = gate_params[gate_name]

        if len(params) != expected_count:
            raise ValidationError(
                f"Gate {gate_name} expects {expected_count} parameters, got {len(params)}"
            )

        validated = []
        for i, param in enumerate(params):
            if param_type == 'angle':
                validated.append(
                    self.num_validator.validate_angle(param, f"{gate_name}_param_{i}")
                )
            elif param_type == 'probability':
                validated.append(
                    self.num_validator.validate_probability(param, f"{gate_name}_param_{i}")
                )
            else:
                validated.append(
                    self.num_validator.validate_real(param, f"{gate_name}_param_{i}")
                )

        return validated

    def validate_measurement(self, qubit: Any, bit: Any) -> Tuple[int, int]:
        """
        Validate measurement operation.

        Args:
            qubit: Qubit to measure
            bit: Classical bit to store result

        Returns:
            Tuple of (qubit_index, bit_index)

        Raises:
            ValidationError: If validation fails
        """
        qubit_idx = IndexValidator.validate_qubit_index(
            qubit, self.num_qubits, "measurement_qubit"
        )
        bit_idx = IndexValidator.validate_bit_index(
            bit, self.num_bits, "measurement_bit"
        )

        return qubit_idx, bit_idx


def create_validator(circuit: Any) -> CircuitValidator:
    """
    Create a validator for a circuit.

    Args:
        circuit: Quantum circuit object

    Returns:
        Configured CircuitValidator
    """
    if hasattr(circuit, 'n_qubits'):
        num_qubits = circuit.n_qubits
    elif hasattr(circuit, 'num_qubits'):
        num_qubits = circuit.num_qubits
    else:
        raise ValidationError("Cannot determine number of qubits in circuit")

    num_bits = getattr(circuit, 'n_bits', None) or getattr(circuit, 'num_bits', num_qubits)

    return CircuitValidator(num_qubits, num_bits)