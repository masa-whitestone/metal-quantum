"""
metalq/result.py - Execution Result Class

回路実行の結果を格納。測定カウント、状態ベクトル、期待値などを提供。

Example:
    result = mq.run(qc, shots=1000)
    print(result.counts)           # {'00': 512, '11': 488}
    print(result.probabilities)    # {'00': 0.512, '11': 0.488}
    print(result.most_frequent())  # '00'
"""
from __future__ import annotations
from typing import Dict, Optional, List, Union
import numpy as np


class Result:
    """
    Result of quantum circuit execution.
    
    Attributes:
        counts: Measurement counts (bitstring -> count)
        statevector: Full statevector (if shots=0)
        expectation: Expectation value (if observable specified)
    """
    
    __slots__ = ('_counts', '_statevector', '_expectation', '_metadata')
    
    def __init__(self,
                 counts: Optional[Dict[str, int]] = None,
                 statevector: Optional[np.ndarray] = None,
                 expectation: Optional[float] = None,
                 metadata: Optional[Dict] = None):
        """
        Initialize result.
        
        Args:
            counts: Measurement counts
            statevector: State vector (complex array)
            expectation: Expectation value
            metadata: Additional metadata (timing, backend info, etc.)
        """
        self._counts = counts or {}
        self._statevector = statevector
        self._expectation = expectation
        self._metadata = metadata or {}
    
    # ========================================================================
    # Properties
    # ========================================================================
    
    @property
    def counts(self) -> Dict[str, int]:
        """Get measurement counts (copy)."""
        return self._counts.copy()
    
    @property
    def statevector(self) -> Optional[np.ndarray]:
        """Get statevector (if available)."""
        if self._statevector is None:
            return None
        return self._statevector.copy()
    
    @property
    def expectation(self) -> Optional[float]:
        """Get expectation value (if computed)."""
        return self._expectation
    
    @property
    def metadata(self) -> Dict:
        """Get execution metadata."""
        return self._metadata.copy()
    
    @property
    def total_counts(self) -> int:
        """Get total number of shots."""
        return sum(self._counts.values())
    
    @property
    def probabilities(self) -> Dict[str, float]:
        """
        Get probability distribution from counts.
        
        Returns:
            Dict mapping bitstrings to probabilities
        """
        total = self.total_counts
        if total == 0:
            return {}
        return {k: v / total for k, v in self._counts.items()}
    
    # ========================================================================
    # Analysis Methods
    # ========================================================================
    
    def most_frequent(self) -> str:
        """
        Get the most frequently measured bitstring.
        
        Returns:
            Most common measurement result
            
        Raises:
            ValueError: If no measurements available
        """
        if not self._counts:
            raise ValueError("No measurement results available")
        return max(self._counts, key=self._counts.get)
    
    def get_counts(self, num_bits: Optional[int] = None) -> Dict[str, int]:
        """
        Get counts with optional bit padding.
        
        Args:
            num_bits: Pad bitstrings to this length
            
        Returns:
            Measurement counts
        """
        if num_bits is None:
            return self.counts
        
        return {
            k.zfill(num_bits): v 
            for k, v in self._counts.items()
        }
    
    def probability(self, bitstring: str) -> float:
        """
        Get probability of a specific outcome.
        
        Args:
            bitstring: Measurement outcome
            
        Returns:
            Probability (0 if not observed)
        """
        total = self.total_counts
        if total == 0:
            return 0.0
        return self._counts.get(bitstring, 0) / total
    
    def expectation_from_counts(self, observable_fn) -> float:
        """
        Compute expectation value from counts using a function.
        
        Args:
            observable_fn: Function mapping bitstring to observable value
            
        Returns:
            Expectation value
            
        Example:
            # Compute <Z₀> from counts
            exp_z0 = result.expectation_from_counts(
                lambda s: 1 if s[-1] == '0' else -1
            )
        """
        total = self.total_counts
        if total == 0:
            return 0.0
        
        return sum(
            count * observable_fn(bitstring)
            for bitstring, count in self._counts.items()
        ) / total
    
    def marginal(self, qubits: List[int]) -> Dict[str, int]:
        """
        Get marginal distribution over specified qubits.
        
        Args:
            qubits: List of qubit indices to keep
            
        Returns:
            Marginal counts
            
        Example:
            # Get counts for qubits 0 and 2 only
            marginal = result.marginal([0, 2])
        """
        if not self._counts:
            return {}
        
        # Determine total number of qubits from bitstring length
        sample_key = next(iter(self._counts))
        n_qubits = len(sample_key)
        
        marginal_counts = {}
        for bitstring, count in self._counts.items():
            # Extract specified qubits (bitstring is big-endian: q_{n-1}...q_1 q_0)
            marginal_bits = ''.join(bitstring[n_qubits - 1 - q] for q in sorted(qubits, reverse=True))
            marginal_counts[marginal_bits] = marginal_counts.get(marginal_bits, 0) + count
        
        return marginal_counts
    
    # ========================================================================
    # Statevector Methods
    # ========================================================================
    
    def amplitudes(self) -> Optional[Dict[str, complex]]:
        """
        Get amplitudes as dict (bitstring -> complex amplitude).
        
        Returns:
            Dict mapping bitstrings to amplitudes, or None if no statevector
        """
        if self._statevector is None:
            return None
        
        n_qubits = int(np.log2(len(self._statevector)))
        return {
            format(i, f'0{n_qubits}b'): self._statevector[i]
            for i in range(len(self._statevector))
        }
    
    def probabilities_from_statevector(self) -> Optional[Dict[str, float]]:
        """
        Get probabilities from statevector.
        
        Returns:
            Dict mapping bitstrings to probabilities
        """
        if self._statevector is None:
            return None
        
        n_qubits = int(np.log2(len(self._statevector)))
        probs = np.abs(self._statevector) ** 2
        
        return {
            format(i, f'0{n_qubits}b'): probs[i]
            for i in range(len(probs))
            if probs[i] > 1e-10  # Filter near-zero probabilities
        }
    
    # ========================================================================
    # String Representation
    # ========================================================================
    
    def __repr__(self) -> str:
        parts = []
        if self._counts:
            parts.append(f"counts={self._counts}")
        if self._statevector is not None:
            parts.append(f"statevector[{len(self._statevector)}]")
        if self._expectation is not None:
            parts.append(f"expectation={self._expectation:.6f}")
        
        return f"Result({', '.join(parts)})"
    
    def __str__(self) -> str:
        if self._counts:
            return str(self._counts)
        elif self._statevector is not None:
            return f"Statevector({len(self._statevector)} amplitudes)"
        elif self._expectation is not None:
            return f"Expectation: {self._expectation:.6f}"
        else:
            return "Result(empty)"
