"""
benchmarks/benchmark_mps.py - Metal-Q Performance Benchmark
"""
import time
import numpy as np
from metalq import Circuit
from metalq.backends.mps.backend import MPSBackend
from metalq.backends.cpu.backend import CPUBackend

def benchmark_circuit(num_qubits, depth, shots=0):
    print(f"Benchmarking {num_qubits} qubits, depth {depth}...")
    
    # Create random circuit
    qc = Circuit(num_qubits)
    for _ in range(depth):
        for i in range(num_qubits):
            qc.h(i)
            qc.rx(0.5, i)
        for i in range(num_qubits - 1):
            qc.cx(i, i+1)
            
    # CPU
    cpu = CPUBackend()
    start = time.perf_counter()
    cpu.run(qc, shots=shots)
    cpu_time = time.perf_counter() - start
    print(f"CPU Time: {cpu_time*1000:.2f} ms")
    
    # MPS
    try:
        mps = MPSBackend()
        start = time.perf_counter()
        mps.run(qc, shots=shots)
        mps_time = time.perf_counter() - start
        print(f"MPS Time: {mps_time*1000:.2f} ms")
        
        speedup = cpu_time / mps_time
        print(f"Speedup: {speedup:.2f}x")
    except Exception as e:
        print(f"MPS Failed: {e}")

if __name__ == "__main__":
    print("--- Metal-Q Benchmark ---")
    
    # Small scale (warmup)
    benchmark_circuit(4, 10)
    
    # Medium scale
    benchmark_circuit(15, 20)
    
    # Large scale (if supported)
    # benchmark_circuit(20, 50) 
