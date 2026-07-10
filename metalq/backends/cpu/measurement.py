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
                  use_polars: bool = True,
                  index_map=None) -> Dict[str, int]:
    """
    Sample measurement results from statevector.

    Args:
        sv: Complex statevector (complex64 or complex128)
        shots: Number of shots
        num_qubits: Number of qubits
        use_polars: Use Polars for aggregation (faster for large shots)
        index_map: Optional bit layout (fusion の lazy permutation)。
            index_map[q] = qubit q の状態が載っているインデックスビット。
            サンプルしたインデックスのビットだけを論理順へ戻す
            (statevector 全体の復元パスより桁違いに安い)。

    Returns:
        Dict mapping bitstrings to counts
    """
    # Compute probabilities in float64 regardless of state dtype
    # (complex64 の確率を float32 で持つと正規化誤差が出る)
    probs = np.square(sv.real, dtype=np.float64)
    probs += np.square(sv.imag, dtype=np.float64)

    # Normalize (handle numerical errors). In-place: probs is our own
    # freshly-built float64 array. np.random.choice tolerates ~sqrt(eps)
    # deviation, so 1e-10 avoids a needless full-array pass for fp64
    # rounding noise while still fixing complex64 states (~1e-5 off).
    probs_sum = probs.sum(dtype=np.float64)
    if abs(probs_sum - 1.0) > 1e-10:
        probs /= probs_sum

    # Sample indices
    indices = np.random.choice(len(sv), size=shots, p=probs)

    # Remap sampled index bits from the permuted layout to logical order
    if index_map is not None and list(index_map) != list(range(num_qubits)):
        remapped = np.zeros_like(indices)
        for q, p in enumerate(index_map):
            remapped |= ((indices >> p) & 1) << q
        indices = remapped

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
