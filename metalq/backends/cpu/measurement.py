"""
metalq/backends/cpu/measurement.py - Measurement and Sampling

Polars による並列集計で測定結果処理を高速化。
"""
import numpy as np
from typing import Dict

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def sample_counts(sv: np.ndarray, shots: int, num_qubits: int,
                  use_polars: bool = True) -> Dict[str, int]:
    """
    Sample measurement results from statevector.
    
    Args:
        sv: Complex statevector
        shots: Number of shots
        num_qubits: Number of qubits
        use_polars: Use Polars for aggregation (faster for large shots)
    
    Returns:
        Dict mapping bitstrings to counts
    """
    # Compute probabilities
    probs = np.abs(sv) ** 2
    
    # Normalize (handle numerical errors)
    probs_sum = probs.sum()
    if abs(probs_sum - 1.0) > 1e-10:
        probs = probs / probs_sum
    
    # Sample indices
    indices = np.random.choice(len(sv), size=shots, p=probs)
    
    if use_polars and HAS_POLARS and shots >= 1000:
        # Use Polars for fast aggregation
        return _aggregate_with_polars(indices, num_qubits)
    else:
        # Pure Python aggregation
        return _aggregate_python(indices, num_qubits)


def _aggregate_with_polars(indices: np.ndarray, num_qubits: int) -> Dict[str, int]:
    """
    Aggregate sample indices using Polars.
    
    Polars の並列処理で大量の shots を高速に集計。
    """
    # Create DataFrame
    df = pl.DataFrame({'idx': indices})
    
    # Group by and count
    counts_df = df.group_by('idx').agg(pl.len().alias('count'))
    
    # Convert to dict with bitstrings
    result = {}
    for row in counts_df.iter_rows():
        idx, count = row
        bitstring = format(idx, f'0{num_qubits}b')
        result[bitstring] = count
    
    return result


def _aggregate_python(indices: np.ndarray, num_qubits: int) -> Dict[str, int]:
    """Aggregate using pure Python (for small shots or no Polars)."""
    counts = {}
    for idx in indices:
        bitstring = format(idx, f'0{num_qubits}b')
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts


def sample_with_memory_efficiency(sv: np.ndarray, shots: int, 
                                   num_qubits: int, 
                                   chunk_size: int = 100000) -> Dict[str, int]:
    """
    Memory-efficient sampling for very large shots.
    
    大量の shots をチャンクに分けて処理し、メモリ使用量を抑える。
    """
    probs = np.abs(sv) ** 2
    probs = probs / probs.sum()
    
    all_counts = {}
    remaining = shots
    
    while remaining > 0:
        chunk = min(chunk_size, remaining)
        indices = np.random.choice(len(sv), size=chunk, p=probs)
        
        chunk_counts = _aggregate_python(indices, num_qubits)
        
        for bitstring, count in chunk_counts.items():
            all_counts[bitstring] = all_counts.get(bitstring, 0) + count
        
        remaining -= chunk
    
    return all_counts
