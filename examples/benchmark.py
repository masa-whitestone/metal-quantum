"""
benchmark.py - Performance comparison: Metal-Q vs Qiskit Aer

Compares execution time for various circuit sizes and types.
"""

import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import metalq


def benchmark_statevector(num_qubits: int, depth: int = 10) -> dict:
    """Benchmark statevector simulation for random circuit."""
    
    # Create random circuit
    qc = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for q in range(num_qubits):
            qc.h(q)
            qc.rz(np.random.uniform(0, 2*np.pi), q)
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
    
    # Metal-Q
    start = time.perf_counter()
    sv_metalq = metalq.statevector(qc)
    metalq_time = time.perf_counter() - start
    
    # Qiskit Statevector
    start = time.perf_counter()
    sv_qiskit = Statevector(qc).data
    qiskit_time = time.perf_counter() - start
    
    # Verify correctness
    match = np.allclose(np.abs(sv_metalq), np.abs(sv_qiskit), atol=1e-4)
    
    return {
        'qubits': num_qubits,
        'depth': depth,
        'metalq_ms': metalq_time * 1000,
        'qiskit_ms': qiskit_time * 1000,
        'speedup': qiskit_time / metalq_time if metalq_time > 0 else 0,
        'match': match
    }


def benchmark_shots(num_qubits: int, shots: int = 1024) -> dict:
    """Benchmark sampling with measurements."""
    
    # GHZ circuit
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(0)
    for q in range(num_qubits - 1):
        qc.cx(q, q + 1)
    qc.measure(range(num_qubits), range(num_qubits))
    
    # Metal-Q
    start = time.perf_counter()
    result_metalq = metalq.run(qc, shots=shots)
    metalq_time = time.perf_counter() - start
    
    # Qiskit Aer
    backend = AerSimulator()
    start = time.perf_counter()
    job = backend.run(qc, shots=shots)
    result_qiskit = job.result()
    qiskit_time = time.perf_counter() - start
    
    return {
        'qubits': num_qubits,
        'shots': shots,
        'metalq_ms': metalq_time * 1000,
        'qiskit_aer_ms': qiskit_time * 1000,
        'speedup': qiskit_time / metalq_time if metalq_time > 0 else 0
    }


def run_benchmarks():
    print("=" * 60)
    print("Metal-Q vs Qiskit Performance Benchmark")
    print("=" * 60)
    print()
    
    # Statevector benchmarks
    print("📊 Statevector Simulation (depth=10)")
    print("-" * 60)
    print(f"{'Qubits':>8} | {'Metal-Q (ms)':>12} | {'Qiskit (ms)':>12} | {'Speedup':>8} | {'Match'}")
    print("-" * 60)
    
    for n in [4, 8, 12, 16, 20]:
        try:
            result = benchmark_statevector(n, depth=10)
            print(f"{result['qubits']:>8} | {result['metalq_ms']:>12.2f} | {result['qiskit_ms']:>12.2f} | {result['speedup']:>7.2f}x | {'✅' if result['match'] else '❌'}")
        except Exception as e:
            print(f"{n:>8} | Error: {e}")
    
    print()
    
    # Sampling benchmarks - standard shots
    print("📊 Sampling (shots=1024)")
    print("-" * 60)
    print(f"{'Qubits':>8} | {'Metal-Q (ms)':>12} | {'Aer (ms)':>12} | {'Speedup':>8}")
    print("-" * 60)
    
    for n in [4, 8, 12, 16, 20]:
        try:
            result = benchmark_shots(n, shots=1024)
            print(f"{result['qubits']:>8} | {result['metalq_ms']:>12.2f} | {result['qiskit_aer_ms']:>12.2f} | {result['speedup']:>7.2f}x")
        except Exception as e:
            print(f"{n:>8} | Error: {e}")
    
    print()
    
    # Sampling benchmarks - high shots
    print("📊 Sampling (shots=8192)")
    print("-" * 60)
    print(f"{'Qubits':>8} | {'Metal-Q (ms)':>12} | {'Aer (ms)':>12} | {'Speedup':>8}")
    print("-" * 60)
    
    for n in [4, 12, 16, 20]:
        try:
            result = benchmark_shots(n, shots=8192)
            print(f"{result['qubits']:>8} | {result['metalq_ms']:>12.2f} | {result['qiskit_aer_ms']:>12.2f} | {result['speedup']:>7.2f}x")
        except Exception as e:
            print(f"{n:>8} | Error: {e}")
    
    print()
    print("=" * 60)
    print("Benchmark complete!")


if __name__ == "__main__":
    run_benchmarks()
