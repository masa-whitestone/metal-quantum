"""
metalq/spin.py - Pauli Spin Operators for Hamiltonian Construction

CUDA-Qライクな構文でハミルトニアンを構築。

Example:
    from metalq.spin import X, Y, Z, I
    
    # 簡単なハミルトニアン
    H = Z(0)
    
    # テンソル積
    H = Z(0) @ Z(1)
    
    # 係数付き
    H = -1.0 * Z(0) @ Z(1) + 0.5 * X(0) + 0.5 * X(1)
    
    # 期待値計算
    energy = mq.expect(qc, H)
"""
from __future__ import annotations
from typing import List, Tuple, Union, Optional, Set
from dataclasses import dataclass, field
import numpy as np


# Pauli matrices (for matrix conversion)
_PAULI_MATRICES = {
    'I': np.array([[1, 0], [0, 1]], dtype=np.complex128),
    'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


@dataclass
class PauliTerm:
    """
    A single Pauli term: coefficient * (P₀ ⊗ P₁ ⊗ ... ⊗ Pₙ)
    
    Attributes:
        coeff: Complex coefficient
        ops: List of (pauli_type, qubit_index) tuples
    
    Example:
        term = PauliTerm(1.0, [('Z', 0), ('Z', 1)])  # Z₀ ⊗ Z₁
    """
    coeff: complex = 1.0
    ops: List[Tuple[str, int]] = field(default_factory=list)
    
    def __post_init__(self):
        self.coeff = complex(self.coeff)
    
    @property
    def qubits(self) -> Set[int]:
        """Get set of qubits this term acts on."""
        return {q for _, q in self.ops}
    
    @property
    def num_qubits(self) -> int:
        """Infer minimum number of qubits needed."""
        if not self.ops:
            return 0
        return max(q for _, q in self.ops) + 1
    
    def __repr__(self) -> str:
        if not self.ops:
            return f"{self.coeff}"
        
        ops_str = " @ ".join(f"{p}({q})" for p, q in self.ops)
        
        if self.coeff == 1:
            return ops_str
        elif self.coeff == -1:
            return f"-{ops_str}"
        elif self.coeff.imag == 0:
            return f"{self.coeff.real} * {ops_str}"
        else:
            return f"{self.coeff} * {ops_str}"
    
    def __hash__(self) -> int:
        return hash((self.coeff, tuple(self.ops)))
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, PauliTerm):
            return self.coeff == other.coeff and self.ops == other.ops
        return False
    
    # ========================================================================
    # Arithmetic Operations
    # ========================================================================
    
    def __neg__(self) -> PauliTerm:
        return PauliTerm(-self.coeff, self.ops.copy())
    
    def __mul__(self, other: Union[int, float, complex, PauliTerm]) -> Union[PauliTerm, 'Hamiltonian']:
        if isinstance(other, (int, float, complex)):
            return PauliTerm(self.coeff * other, self.ops.copy())
        elif isinstance(other, PauliTerm):
            # Tensor product of two terms
            return PauliTerm(
                self.coeff * other.coeff,
                self.ops + other.ops
            )
        return NotImplemented
    
    def __rmul__(self, other: Union[int, float, complex]) -> PauliTerm:
        if isinstance(other, (int, float, complex)):
            return PauliTerm(other * self.coeff, self.ops.copy())
        return NotImplemented
    
    def __matmul__(self, other: PauliTerm) -> PauliTerm:
        """Tensor product using @ operator."""
        if isinstance(other, PauliTerm):
            return PauliTerm(
                self.coeff * other.coeff,
                self.ops + other.ops
            )
        return NotImplemented
    
    def __truediv__(self, other: Union[int, float, complex]) -> PauliTerm:
        if isinstance(other, (int, float, complex)):
            return PauliTerm(self.coeff / other, self.ops.copy())
        return NotImplemented
    
    def __add__(self, other: Union[int, float, complex, PauliTerm, 'Hamiltonian']) -> 'Hamiltonian':
        if isinstance(other, (int, float, complex)):
            return Hamiltonian([self, PauliTerm(other, [])])
        elif isinstance(other, PauliTerm):
            return Hamiltonian([self, other])
        elif isinstance(other, Hamiltonian):
            return Hamiltonian([self] + other.terms)
        return NotImplemented
    
    def __radd__(self, other: Union[int, float, complex, PauliTerm]) -> 'Hamiltonian':
        if isinstance(other, (int, float, complex)):
            return Hamiltonian([PauliTerm(other, []), self])
        return self.__add__(other)
    
    def __sub__(self, other: Union[int, float, complex, PauliTerm, 'Hamiltonian']) -> 'Hamiltonian':
        if isinstance(other, (int, float, complex)):
            return Hamiltonian([self, PauliTerm(-other, [])])
        elif isinstance(other, PauliTerm):
            return Hamiltonian([self, -other])
        elif isinstance(other, Hamiltonian):
            return Hamiltonian([self] + [-t for t in other.terms])
        return NotImplemented
    
    def __rsub__(self, other: Union[int, float, complex]) -> 'Hamiltonian':
        if isinstance(other, (int, float, complex)):
            return Hamiltonian([PauliTerm(other, []), -self])
        return NotImplemented
    
    # ========================================================================
    # Matrix Conversion
    # ========================================================================
    
    def to_matrix(self, num_qubits: Optional[int] = None) -> np.ndarray:
        """
        Convert to full matrix representation.
        
        Args:
            num_qubits: Total number of qubits (inferred if not specified)
            
        Returns:
            2^n × 2^n complex matrix
        """
        if num_qubits is None:
            num_qubits = self.num_qubits
        
        if num_qubits == 0:
            return np.array([[self.coeff]], dtype=np.complex128)
        
        # Build operator dict
        qubit_ops = {q: 'I' for q in range(num_qubits)}
        for pauli, qubit in self.ops:
            if qubit in qubit_ops:
                # Multiply Paulis on same qubit
                qubit_ops[qubit] = self._multiply_paulis(qubit_ops[qubit], pauli)
            else:
                qubit_ops[qubit] = pauli
        
        # Tensor product
        matrices = [_PAULI_MATRICES[qubit_ops[q]] for q in range(num_qubits)]
        
        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        
        return self.coeff * result
    
    @staticmethod
    def _multiply_paulis(p1: str, p2: str) -> str:
        """Multiply two Pauli operators (simplified)."""
        if p1 == 'I':
            return p2
        if p2 == 'I':
            return p1
        if p1 == p2:
            return 'I'
        # XY, YZ, ZX cases would need phase tracking
        # For now, return the second one (simplified)
        return p2


class Hamiltonian:
    """
    Sum of Pauli terms: H = Σᵢ cᵢ Pᵢ
    
    Example:
        H = -1.0 * Z(0) @ Z(1) + 0.5 * X(0) + 0.5 * X(1)
        
        # Get matrix form
        matrix = H.to_matrix(num_qubits=2)
        
        # Iterate over terms
        for term in H.terms:
            print(term)
    """
    
    __slots__ = ('_terms',)
    
    def __init__(self, terms: Optional[List[PauliTerm]] = None):
        self._terms = terms or []
    
    @property
    def terms(self) -> List[PauliTerm]:
        """Get list of terms (copy)."""
        return self._terms.copy()
    
    @property
    def num_terms(self) -> int:
        """Get number of terms."""
        return len(self._terms)
    
    @property
    def num_qubits(self) -> int:
        """Infer minimum number of qubits needed."""
        if not self._terms:
            return 0
        return max(t.num_qubits for t in self._terms)
    
    @property
    def qubits(self) -> Set[int]:
        """Get set of all qubits involved."""
        result = set()
        for term in self._terms:
            result.update(term.qubits)
        return result
    
    def __repr__(self) -> str:
        if not self._terms:
            return "0"
        
        parts = []
        for i, term in enumerate(self._terms):
            term_str = str(term)
            if i == 0:
                parts.append(term_str)
            elif term_str.startswith('-'):
                parts.append(f" - {term_str[1:]}")
            else:
                parts.append(f" + {term_str}")
        
        return ''.join(parts)
    
    def __iter__(self):
        return iter(self._terms)
    
    def __len__(self) -> int:
        return len(self._terms)
    
    # ========================================================================
    # Arithmetic Operations
    # ========================================================================
    
    def __neg__(self) -> Hamiltonian:
        return Hamiltonian([-t for t in self._terms])
    
    def __add__(self, other: Union[int, float, complex, PauliTerm, Hamiltonian]) -> Hamiltonian:
        if isinstance(other, (int, float, complex)):
            return Hamiltonian(self._terms + [PauliTerm(other, [])])
        elif isinstance(other, PauliTerm):
            return Hamiltonian(self._terms + [other])
        elif isinstance(other, Hamiltonian):
            return Hamiltonian(self._terms + other._terms)
        return NotImplemented
    
    def __radd__(self, other: Union[int, float, complex, PauliTerm]) -> Hamiltonian:
        return self.__add__(other)
    
    def __sub__(self, other: Union[int, float, complex, PauliTerm, Hamiltonian]) -> Hamiltonian:
        if isinstance(other, (int, float, complex)):
            return Hamiltonian(self._terms + [PauliTerm(-other, [])])
        elif isinstance(other, PauliTerm):
            return Hamiltonian(self._terms + [-other])
        elif isinstance(other, Hamiltonian):
            return Hamiltonian(self._terms + [-t for t in other._terms])
        return NotImplemented
    
    def __rsub__(self, other: Union[int, float, complex]) -> Hamiltonian:
        return Hamiltonian([PauliTerm(other, [])] + [-t for t in self._terms])
    
    def __mul__(self, scalar: Union[int, float, complex]) -> Hamiltonian:
        if isinstance(scalar, (int, float, complex)):
            return Hamiltonian([
                PauliTerm(t.coeff * scalar, t.ops) 
                for t in self._terms
            ])
        return NotImplemented
    
    def __rmul__(self, scalar: Union[int, float, complex]) -> Hamiltonian:
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: Union[int, float, complex]) -> Hamiltonian:
        if isinstance(scalar, (int, float, complex)):
            return Hamiltonian([
                PauliTerm(t.coeff / scalar, t.ops) 
                for t in self._terms
            ])
        return NotImplemented
    
    # ========================================================================
    # Simplification
    # ========================================================================
    
    def simplify(self) -> Hamiltonian:
        """
        Combine like terms.
        
        Returns:
            New Hamiltonian with combined terms
        """
        term_dict = {}
        
        for term in self._terms:
            # Create a hashable key from operators
            key = tuple(sorted(term.ops))
            
            if key in term_dict:
                term_dict[key] = PauliTerm(
                    term_dict[key].coeff + term.coeff,
                    list(key)
                )
            else:
                term_dict[key] = PauliTerm(term.coeff, list(key))
        
        # Filter out zero terms
        simplified = [t for t in term_dict.values() if abs(t.coeff) > 1e-15]
        
        return Hamiltonian(simplified)
    
    # ========================================================================
    # Matrix Conversion
    # ========================================================================
    
    def to_matrix(self, num_qubits: Optional[int] = None) -> np.ndarray:
        """
        Convert to full matrix representation.
        
        Args:
            num_qubits: Total number of qubits (inferred if not specified)
            
        Returns:
            2^n × 2^n complex Hermitian matrix
        """
        if num_qubits is None:
            num_qubits = self.num_qubits
        
        if num_qubits == 0:
            # Constant term only
            const = sum(t.coeff for t in self._terms if not t.ops)
            return np.array([[const]], dtype=np.complex128)
        
        dim = 2 ** num_qubits
        result = np.zeros((dim, dim), dtype=np.complex128)
        
        for term in self._terms:
            result += term.to_matrix(num_qubits)
        
        return result
    
    def to_sparse(self, num_qubits: Optional[int] = None):
        """
        Convert to sparse matrix representation.
        
        Returns:
            scipy.sparse matrix
        """
        from scipy import sparse
        return sparse.csr_matrix(self.to_matrix(num_qubits))
    
    # ========================================================================
    # Serialization
    # ========================================================================
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'terms': [
                {
                    'coeff_real': t.coeff.real,
                    'coeff_imag': t.coeff.imag,
                    'ops': t.ops
                }
                for t in self._terms
            ]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> Hamiltonian:
        """Create from dictionary."""
        terms = [
            PauliTerm(
                complex(t['coeff_real'], t['coeff_imag']),
                [tuple(op) for op in t['ops']]
            )
            for t in data['terms']
        ]
        return cls(terms)


# ============================================================================
# Convenience Factory Functions
# ============================================================================

def X(qubit: int) -> PauliTerm:
    """Create Pauli-X operator on specified qubit."""
    return PauliTerm(1.0, [('X', qubit)])

def Y(qubit: int) -> PauliTerm:
    """Create Pauli-Y operator on specified qubit."""
    return PauliTerm(1.0, [('Y', qubit)])

def Z(qubit: int) -> PauliTerm:
    """Create Pauli-Z operator on specified qubit."""
    return PauliTerm(1.0, [('Z', qubit)])

def I(qubit: int) -> PauliTerm:
    """Create Identity operator on specified qubit."""
    return PauliTerm(1.0, [('I', qubit)])


# Aliases
Sx = X
Sy = Y
Sz = Z
Si = I
