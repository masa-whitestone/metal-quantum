#!/usr/bin/env python3
"""
large_scale_benchmark.py - Large-scale Quantum Circuit Benchmarks

Benchmarks Metal-Q against Qiskit Aer for 16-28 qubit circuits.
"""

import json
import time
import sys
import gc
from datetime import datetime
from typing import Dict, Any

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


def create_random_circuit(num_qubits: int, depth: int, seed: int = 42) -> QuantumCircuit:
    """Create a random quantum circuit."""
    np.random.seed(seed)
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    for d in range(depth):
        for q in range(num_qubits):
            gate = np.random.choice(['h', 'x', 't', 's', 'rx', 'ry', 'rz'])
            if gate == 'h': qc.h(q)
            elif gate == 'x': qc.x(q)
            elif gate == 't': qc.t(q)
            elif gate == 's': qc.s(q)
            elif gate == 'rx': qc.rx(np.random.uniform(0, 2*np.pi), q)
            elif gate == 'ry': qc.ry(np.random.uniform(0, 2*np.pi), q)
            elif gate == 'rz': qc.rz(np.random.uniform(0, 2*np.pi), q)
        
        for q in range(0, num_qubits - 1, 2):
            qc.cx(q, q + 1)
        for q in range(1, num_qubits - 1, 2):
            qc.cx(q, q + 1)
    
    return qc


def create_qft_circuit(num_qubits: int) -> QuantumCircuit:
    """Create a QFT circuit."""
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    for i in range(num_qubits):
        qc.h(i)
        for j in range(i + 1, num_qubits):
            angle = np.pi / (2 ** (j - i))
            qc.cp(angle, j, i)
    
    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - 1 - i)
    
    return qc


def benchmark_statevector(qc: QuantumCircuit, num_runs: int = 3) -> Dict[str, float]:
    """Benchmark statevector simulation."""
    results = {'metalq': None, 'qiskit': None}
    
    # Warm up Metal-Q
    try:
        _ = metalq.statevector(qc)
    except Exception as e:
        print(f"  Metal-Q warm-up failed: {e}")
        return results
    
    # Benchmark Metal-Q
    gc.collect()
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = metalq.statevector(qc)
        times.append(time.perf_counter() - start)
    results['metalq'] = np.median(times) * 1000
    
    # Benchmark Qiskit
    gc.collect()
    times = []
    try:
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = Statevector.from_instruction(qc)
            times.append(time.perf_counter() - start)
        results['qiskit'] = np.median(times) * 1000
    except Exception as e:
        print(f"  Qiskit failed: {e}")
    
    return results


def benchmark_sampling(qc: QuantumCircuit, shots: int, num_runs: int = 3) -> Dict[str, float]:
    """Benchmark sampling."""
    results = {'metalq': None, 'aer': None}
    
    qc_meas = qc.copy()
    qc_meas.measure(range(qc.num_qubits), range(qc.num_qubits))
    
    # Warm up
    try:
        _ = metalq.run(qc_meas, shots=shots)
    except Exception as e:
        print(f"  Metal-Q warm-up failed: {e}")
        return results
    
    # Metal-Q
    gc.collect()
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = metalq.run(qc_meas, shots=shots)
        times.append(time.perf_counter() - start)
    results['metalq'] = np.median(times) * 1000
    
    # Aer
    if HAS_AER:
        gc.collect()
        times = []
        try:
            backend = AerSimulator(method='statevector')
            for _ in range(num_runs):
                start = time.perf_counter()
                job = backend.run(qc_meas, shots=shots)
                _ = job.result()
                times.append(time.perf_counter() - start)
            results['aer'] = np.median(times) * 1000
        except Exception as e:
            print(f"  Aer failed: {e}")
    
    return results


def run_benchmarks() -> Dict[str, Any]:
    """Run all benchmarks."""
    results = {
        'metadata': {'date': datetime.now().isoformat()},
        'statevector': [],
        'sampling': [],
        'qft': [],
    }
    
    # Statevector benchmarks
    print("\n" + "=" * 60)
    print("STATEVECTOR BENCHMARKS")
    print("=" * 60)
    
    sv_configs = [(16, 10), (20, 10), (22, 10), (24, 8), (26, 6)]
    
    for num_qubits, depth in sv_configs:
        print(f"\n{num_qubits} qubits, depth {depth}...")
        qc = create_random_circuit(num_qubits, depth)
        bench = benchmark_statevector(qc)
        
        entry = {
            'num_qubits': num_qubits,
            'depth': depth,
            'metalq_ms': bench['metalq'],
            'qiskit_ms': bench['qiskit'],
            'speedup': bench['qiskit'] / bench['metalq'] if bench['metalq'] and bench['qiskit'] else None
        }
        
        if entry['speedup']:
            print(f"  Metal-Q: {bench['metalq']:.0f}ms, Qiskit: {bench['qiskit']:.0f}ms, Speedup: {entry['speedup']:.1f}x")
        results['statevector'].append(entry)
        gc.collect()
    
    # Sampling benchmarks
    print("\n" + "=" * 60)
    print("SAMPLING BENCHMARKS")
    print("=" * 60)
    
    sampling_configs = [(16, 10, 8192), (20, 10, 8192), (22, 10, 8192), (24, 8, 4096)]
    
    for num_qubits, depth, shots in sampling_configs:
        print(f"\n{num_qubits} qubits, depth {depth}, {shots} shots...")
        qc = create_random_circuit(num_qubits, depth)
        bench = benchmark_sampling(qc, shots)
        
        entry = {
            'num_qubits': num_qubits,
            'depth': depth,
            'shots': shots,
            'metalq_ms': bench['metalq'],
            'aer_ms': bench['aer'],
            'speedup': bench['aer'] / bench['metalq'] if bench['metalq'] and bench['aer'] else None
        }
        
        if entry['speedup']:
            print(f"  Metal-Q: {bench['metalq']:.0f}ms, Aer: {bench['aer']:.0f}ms, Speedup: {entry['speedup']:.1f}x")
        results['sampling'].append(entry)
        gc.collect()
    
    # QFT benchmarks
    print("\n" + "=" * 60)
    print("QFT BENCHMARKS")
    print("=" * 60)
    
    for num_qubits in [16, 20, 22, 24]:
        print(f"\nQFT {num_qubits} qubits...")
        qc = create_qft_circuit(num_qubits)
        bench = benchmark_statevector(qc)
        
        entry = {
            'num_qubits': num_qubits,
            'metalq_ms': bench['metalq'],
            'qiskit_ms': bench['qiskit'],
            'speedup': bench['qiskit'] / bench['metalq'] if bench['metalq'] and bench['qiskit'] else None
        }
        
        if entry['speedup']:
            print(f"  Metal-Q: {bench['metalq']:.0f}ms, Qiskit: {bench['qiskit']:.0f}ms, Speedup: {entry['speedup']:.1f}x")
        results['qft'].append(entry)
        gc.collect()
    
    return results


def print_markdown(results: Dict[str, Any]):
    """Print Markdown tables."""
    print("\n\n" + "=" * 60)
    print("MARKDOWN FOR README")
    print("=" * 60)
    
    print("\n### Statevector Simulation\n")
    print("| Qubits | Depth | Metal-Q | Qiskit | Speedup |")
    print("|--------|-------|---------|--------|---------|")
    for e in results['statevector']:
        mq = f"{e['metalq_ms']:.0f}ms" if e['metalq_ms'] else "N/A"
        qk = f"{e['qiskit_ms']:.0f}ms" if e['qiskit_ms'] else "OOM"
        sp = f"**{e['speedup']:.1f}x**" if e['speedup'] else "N/A"
        print(f"| {e['num_qubits']} | {e['depth']} | {mq} | {qk} | {sp} |")
    
    print("\n### Sampling (shots=8192)\n")
    print("| Qubits | Metal-Q | Aer | Speedup |")
    print("|--------|---------|-----|---------|")
    for e in results['sampling']:
        mq = f"{e['metalq_ms']:.0f}ms" if e['metalq_ms'] else "N/A"
        aer = f"{e['aer_ms']:.0f}ms" if e['aer_ms'] else "OOM"
        sp = f"**{e['speedup']:.1f}x**" if e['speedup'] else "N/A"
        print(f"| {e['num_qubits']} | {mq} | {aer} | {sp} |")
    
    print("\n### QFT Circuit\n")
    print("| Qubits | Metal-Q | Qiskit | Speedup |")
    print("|--------|---------|--------|---------|")
    for e in results['qft']:
        mq = f"{e['metalq_ms']:.0f}ms" if e['metalq_ms'] else "N/A"
        qk = f"{e['qiskit_ms']:.0f}ms" if e['qiskit_ms'] else "OOM"
        sp = f"**{e['speedup']:.1f}x**" if e['speedup'] else "N/A"
        print(f"| {e['num_qubits']} | {mq} | {qk} | {sp} |")


if __name__ == '__main__':
    print("Metal-Q Large-Scale Benchmark")
    print(f"Date: {datetime.now().isoformat()}")
    
    results = run_benchmarks()
    
    with open('benchmarks/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print_markdown(results)
