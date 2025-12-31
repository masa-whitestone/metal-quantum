"""
CPU Backend Benchmark: Metal-Q CPU vs Qiskit Aer

Compares Metal-Q CPU backend with Qiskit Aer for statevector simulation.
"""

import time
import gc
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

try:
    from qiskit_aer import AerSimulator
    HAS_AER = True
except ImportError:
    HAS_AER = False
    print("Warning: qiskit-aer not installed")

import metalq
from metalq.adapters.qiskit_adapter import to_metalq


def create_random_circuit(num_qubits: int, depth: int, seed: int = 42) -> QuantumCircuit:
    """Create random circuit for benchmarking."""
    np.random.seed(seed)
    qc = QuantumCircuit(num_qubits)
    
    for _ in range(depth):
        for qi in range(num_qubits):
            gate = np.random.choice(['h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz'])
            if gate in ['h', 'x', 'y', 'z', 's', 't']:
                getattr(qc, gate)(qi)
            else:
                angle = np.random.uniform(0, 2*np.pi)
                getattr(qc, gate)(angle, qi)
        
        cx_pairs = np.random.randint(0, num_qubits, size=(num_qubits//2, 2))
        for c, t in cx_pairs:
            if c != t:
                qc.cx(c, t)
    
    return qc


def benchmark_cpu_comparison(num_qubits: int, depth: int, num_runs: int = 3):
    """Run CPU backend comparison."""
    print(f"\n{num_qubits} qubits, depth {depth}...")
    
    qc = create_random_circuit(num_qubits, depth)
    mq_qc = to_metalq(qc)
    
    results = {}
    
    # Metal-Q CPU
    try:
        gc.collect()
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = metalq.statevector(mq_qc, backend='cpu')
            times.append(time.perf_counter() - start)
        results['metalq_cpu'] = np.median(times) * 1000
        print(f"  Metal-Q CPU: {results['metalq_cpu']:.1f}ms")
    except Exception as e:
        print(f"  Metal-Q CPU failed: {e}")
        results['metalq_cpu'] = None
    
    # Qiskit Aer
    if HAS_AER:
        try:
            gc.collect()
            backend = AerSimulator(method='statevector')
            
            # Need to save statevector explicitly
            qc_with_save = qc.copy()
            qc_with_save.save_statevector()
            
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                job = backend.run(qc_with_save)
                result = job.result()
                _ = result.data()['statevector']
                times.append(time.perf_counter() - start)
            results['aer'] = np.median(times) * 1000
            print(f"  Qiskit Aer:  {results['aer']:.1f}ms")
        except Exception as e:
            print(f"  Aer failed: {e}")
            results['aer'] = None
    
    # Qiskit Statevector
    try:
        gc.collect()
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = Statevector.from_instruction(qc)
            times.append(time.perf_counter() - start)
        results['qiskit'] = np.median(times) * 1000
        print(f"  Qiskit:      {results['qiskit']:.1f}ms")
    except Exception as e:
        print(f"  Qiskit failed: {e}")
        results['qiskit'] = None
    
    # Print comparison
    if results.get('metalq_cpu') and results.get('aer'):
        speedup = results['aer'] / results['metalq_cpu']
        print(f"  Speedup vs Aer: {speedup:.2f}x")
    
    return results


def main():
    print("==============================================")
    print("Metal-Q CPU vs Qiskit Aer Benchmark")
    print("==============================================")
    
    test_cases = [
        (10, 10),
        (12, 10),
        (14, 8),
        (16, 8),
        (18, 6),
    ]
    
    all_results = []
    
    for num_qubits, depth in test_cases:
        result = benchmark_cpu_comparison(num_qubits, depth)
        result['num_qubits'] = num_qubits
        result['depth'] = depth
        all_results.append(result)
    
    # Print summary table
    print("\n==============================================")
    print("Summary Table")
    print("==============================================")
    print("\n| Qubits | Depth | Metal-Q CPU | Qiskit Aer | Qiskit | Speedup vs Aer |")
    print("|--------|-------|-------------|------------|--------|----------------|")
    
    for r in all_results:
        mq = r.get('metalq_cpu', 0) or 0
        aer = r.get('aer', 0) or 0
        qiskit = r.get('qiskit', 0) or 0
        speedup = aer / mq if (mq > 0 and aer > 0) else 0
        
        mq_str = f"{mq:.0f}ms" if mq > 0 else "N/A"
        aer_str = f"{aer:.0f}ms" if aer > 0 else "N/A"
        qiskit_str = f"{qiskit:.0f}ms" if qiskit > 0 else "N/A"
        speedup_str = f"**{speedup:.2f}x**" if speedup > 0 else "N/A"
        
        print(f"| {r['num_qubits']:6d} | {r['depth']:5d} | {mq_str:11s} | {aer_str:10s} | {qiskit_str:6s} | {speedup_str:14s} |")


if __name__ == "__main__":
    main()
