#!/usr/bin/env python3
"""benchmarks/cpu_vqe_benchmark.py - CPU backend VQE objective benchmark.

Times the four calls a VQE/QAOA optimizer actually makes on the CPU
backend -- statevector(), expectation(), gradient(method='adjoint') and
expectation_and_gradient() -- on a fixed hardware-efficient ansatz and a
transverse-field-free Ising Hamiltonian, and prints them as a markdown
table (min of N repetitions, milliseconds).

Standalone: needs only numpy + metalq (no qiskit, no torch).

    python benchmarks/cpu_vqe_benchmark.py                    # 12,16,18,20
    python benchmarks/cpu_vqe_benchmark.py -n 4,8 --shift     # + param-shift
    python benchmarks/cpu_vqe_benchmark.py --dtype complex64

NOTE (Linux / OpenBLAS): run with OPENBLAS_NUM_THREADS=1. The fused GEMM
path already splits its work across threads, so an additionally threaded
BLAS oversubscribes the cores and inflates every timing here.
"""
import argparse
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metalq import Circuit, Parameter                      # noqa: E402
from metalq.spin import Hamiltonian, PauliTerm             # noqa: E402
from metalq.backends.cpu.backend import CPUBackend         # noqa: E402


# ============================================================================
# Workload: 3-layer RY/RZ + CX-chain ansatz, 2n-1 term Ising Hamiltonian
# ============================================================================

def ansatz(n: int, layers: int = 3):
    """Hardware-efficient ansatz: (RY^n RZ^n CX-chain) x layers."""
    qc = Circuit(n)
    params = []
    for l in range(layers):
        for q in range(n):
            p = Parameter(f'ry{l}_{q}')
            params.append(p)
            qc.ry(p, q)
        for q in range(n):
            p = Parameter(f'rz{l}_{q}')
            params.append(p)
            qc.rz(p, q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
    return qc, params


def ising(n: int) -> Hamiltonian:
    """sum_q Z_q + 0.5 sum_q Z_q Z_{q+1}  (2n-1 terms)."""
    terms = [PauliTerm(1.0, [('Z', q)]) for q in range(n)]
    terms += [PauliTerm(0.5, [('Z', q), ('Z', q + 1)]) for q in range(n - 1)]
    return Hamiltonian(terms)


def timeit(fn, reps: int) -> float:
    """Min-of-`reps` wall time in ms (one warm-up call is discarded)."""
    fn()
    best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


# ============================================================================
# Platform banner
# ============================================================================

def blas_info() -> str:
    """One-liner naming the BLAS numpy is linked against."""
    try:
        cfg = np.__config__.CONFIG      # numpy >= 1.25
        blas = cfg.get('Build Dependencies', {}).get('blas', {})
        name = blas.get('name', '?')
        ver = blas.get('version', '?')
        return f"{name} {ver}"
    except Exception:
        try:
            import io
            buf = io.StringIO()
            np.__config__.show(mode='dicts')
            return buf.getvalue().splitlines()[0] if buf.getvalue() else '?'
        except Exception:
            return '?'


def banner():
    try:
        import numba
        nb = f"numba {numba.__version__}"
    except ImportError:
        nb = "numba absent"
    print(f"platform : {platform.platform()}")
    print(f"cpus     : {os.cpu_count()}   ({nb})")
    print(f"numpy    : {np.__version__}  BLAS: {blas_info()}")
    env = {k: os.environ[k] for k in
           ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS')
           if k in os.environ}
    print(f"env      : {env or '(BLAS/OMP thread limits unset)'}")
    print()


# ============================================================================
# Main
# ============================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-n', '--qubits', default='12,16,18,20',
                    help='comma-separated qubit counts (default 12,16,18,20)')
    ap.add_argument('-d', '--dtype', default='both',
                    choices=('complex128', 'complex64', 'both'),
                    help='statevector precision (default both)')
    ap.add_argument('-r', '--reps', type=int, default=3,
                    help='timed repetitions, min is reported (default 3)')
    ap.add_argument('-l', '--layers', type=int, default=3,
                    help='ansatz layers (default 3)')
    ap.add_argument('--shift', action='store_true',
                    help='also time the parameter-shift gradient (slow: 2 '
                         'circuit evaluations per gate parameter slot)')
    ap.add_argument('--shift-max-qubits', type=int, default=10,
                    help='skip parameter-shift above this n (default 10)')
    ap.add_argument('--max-fused-qubits', type=int, default=4)
    args = ap.parse_args(argv)

    qubit_list = [int(x) for x in args.qubits.split(',') if x.strip()]
    dtypes = ([np.complex128, np.complex64] if args.dtype == 'both'
              else [getattr(np, args.dtype)])

    banner()

    cols = ['dtype', 'n', 'params', 'gates', 'statevector', 'expectation',
            'gradient', 'exp+grad']
    if args.shift:
        cols.append('param-shift')
    print('| ' + ' | '.join(cols) + ' |')
    print('|' + '|'.join(['---'] * len(cols)) + '|')

    for dtype in dtypes:
        backend = CPUBackend(dtype=dtype,
                             max_fused_qubits=args.max_fused_qubits)
        for n in qubit_list:
            qc, ps = ansatz(n, args.layers)
            H = ising(n)
            vals = list(np.random.default_rng(0).uniform(-1, 1, len(ps)))

            t_sv = timeit(lambda: backend.statevector(qc, vals), args.reps)
            t_ex = timeit(lambda: backend.expectation(qc, H, vals), args.reps)
            t_gr = timeit(lambda: backend.gradient(qc, H, vals), args.reps)
            t_eg = timeit(lambda: backend.expectation_and_gradient(qc, H, vals),
                          args.reps)
            row = [np.dtype(dtype).name, str(n), str(len(ps)),
                   str(len(qc._gates)),
                   f'{t_sv:.1f}', f'{t_ex:.1f}', f'{t_gr:.1f}', f'{t_eg:.1f}']
            if args.shift:
                if n <= args.shift_max_qubits:
                    t_ps = timeit(
                        lambda: backend.gradient(qc, H, vals,
                                                 method='parameter_shift'),
                        max(1, args.reps - 2))
                    row.append(f'{t_ps:.1f}')
                else:
                    row.append('skipped')
            print('| ' + ' | '.join(row) + ' |', flush=True)

    print()
    print('All times are milliseconds, min of '
          f'{args.reps} runs after one warm-up.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
