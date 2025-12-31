
import time
import numpy as np
import metalq as mq
from metalq import Circuit, Parameter, Z

def benchmark_gradients():
    print("Benchmarking Gradient Calculation Speed...")
    print(f"{'Qubits':<8} | {'Depth':<6} | {'Params':<8} | {'MPS-Adjoint (s)':<16} | {'MPS-Shift (s)':<16} | {'Speedup':<8}")
    print("-" * 80)
    
    # Test cases: (num_qubits, depth)
    cases = [
        (4, 10),
        (8, 10),
        (12, 10), 
        (16, 10) # 16 qubits is reasonably heavy for Statevector
    ]
    
    for n_qubits, depth in cases:
        # Build Parameterized Circuit
        qc = Circuit(n_qubits)
        params = []
        
        # Random parameterized circuit
        idx = 0
        for _ in range(depth):
            for q in range(n_qubits):
                theta = Parameter(f'p{idx}')
                params.append(theta)
                qc.rx(theta, q)
                qc.rz(theta, q) # Reusing same param for stress test or new? Let's use new.
                idx += 1
            
            # Entanglement
            for q in range(n_qubits - 1):
                qc.cx(q, q+1)
                
        # Create parameter values
        n_params = len(params)
        # Re-bind logic: we actually need unique parameters for strict gradient test.
        # But Circuit.rx(theta) binds the object.
        # Let's fix above to make unique params.
        
        # Re-build strictly unique params
        qc = Circuit(n_qubits)
        param_list = []
        for d_i in range(depth):
            for q in range(n_qubits):
                p = Parameter(f'l{d_i}_q{q}')
                param_list.append(p)
                qc.rx(p, q)
            for q in range(n_qubits - 1):
                qc.cx(q, q+1)
        
        n_params = len(param_list)
        param_vals = np.random.rand(n_params).tolist()
        
        H = mq.Z(0) * mq.Z(1)
        
        # 1. Adjoint Benchmark
        bk_mps = mq.backends.mps.backend.MPSBackend()
        
        start = time.time()
        # Cold run (MPS context creation overhead excluded by placing backend creation outside, 
        # but JIT/Pipeline creation might happen. Let's run once to warm up?)
        # Warmup
        bk_mps.gradient(qc, H, param_vals, method='adjoint')
        
        t0 = time.time()
        grads_adj = bk_mps.gradient(qc, H, param_vals, method='adjoint')
        t_adj = time.time() - t0
        
        # 2. Parameter Shift Benchmark
        # We also run this on MPS backend to isolate "Method" speedup vs "Backend" speedup.
        # Parameter shift requires 2 * N_params evaluations.
        # For N=16, depth=10, params = 160. 320 runs. This is slow.
        # We limit runs for benchmark sake.
        
        if n_qubits > 12:
             print(f"{n_qubits:<8} | {depth:<6} | {n_params:<8} | {t_adj:<16.4f} | {'(skipped)':<16} | {'N/A':<8}")
             continue

        t0 = time.time()
        # Warmup? No, standard run.
        grads_ps = bk_mps.gradient(qc, H, param_vals, method='parameter_shift')
        t_ps = time.time() - t0
        
        speedup = t_ps / t_adj if t_adj > 1e-6 else 0.0
        
        print(f"{n_qubits:<8} | {depth:<6} | {n_params:<8} | {t_adj:<16.4f} | {t_ps:<16.4f} | {speedup:<8.1f}x")

if __name__ == "__main__":
    benchmark_gradients()
