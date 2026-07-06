---
name: metal-gpu
description: Metal GPU programming for Metal-Q's native layer — writing/modifying Metal compute kernels (.metal), Objective-C host code (native/src/*.m), the C API (metalq.h), or the ctypes bridge. Use when adding GPU kernels, debugging wrong GPU results, optimizing GPU performance, or touching anything under native/.
---

# Metal GPU Programming (Metal-Q native layer)

Patterns for the `native/` layer: Metal Compute Shaders + Objective-C host code
+ C API consumed by Python via ctypes. All rules below come from real bugs and
measured benchmarks in this repo — follow them unless you have measured evidence
to the contrary.

## Architecture map

```
metalq/backends/mps/backend.py      ctypes marshaling, timeouts, validation
        │  (loads native/build/libmetalq.dylib — NOT metalq/lib/)
native/include/metalq.h             C API: mq_gate_t, mq_hamiltonian_t, mq_complex_t
native/src/metalq.m                 context lifecycle, dladdr-based .metallib resolution
native/src/state_vector.m           buffer pool (mq_pool_acquire/release), GPU-side statevector init
native/src/gate_executor.m          get_pipeline cache (@synchronized), encode_gate/encode_circuit, 1q fusion, metalq_run
native/src/expectation.m            metalq_expectation (fused Pauli kernel, GPU-resident ⟨ψ|H|ψ⟩)
native/src/adjoint_diff.m           metalq_gradient_adjoint_energy (fused (E, ∇E); gradients incl. P/CP)
native/src/sampling.m               metalq_sample (block-CDF sampling, no statevector readback)
native/src/quantum_gates.metal      apply_gate_single/two/three
native/src/adjoint.metal            pauli_expectation / pauli_inner_product / pauli_accumulate / block_probability_sums
```

## Build & verify loop

```bash
cd native && make clean && make && cd ..          # → native/build/{libmetalq.dylib,default.metallib}
.venv/bin/python -m pytest tests/test_gpu_correctness.py -q   # GPU vs CPU float64 reference
```

- The backend loads from `native/build/`; stale `metalq/lib/` copies are ignored.
- Adding a new `.metal` file requires editing the Makefile: `$(METALLIB)` lists
  source `.metal` files explicitly and links their `.air` outputs into ONE
  `default.metallib`. New `.m` files are picked up by the `$(SRC_DIR)/*.m` wildcard.
- Correctness = match the CPU backend (complex128) within float32 tolerance.
  Scale tolerances with qubit count: ~1e-5 at 16q is fine; a plain float32 dot
  product at 24q can legitimately drift to ~1e-3 (see Precision below).

## Performance model: memory-bandwidth bound

A statevector kernel dispatch reads+writes the full 2^n × 8-byte state. Almost
nothing else matters. Consequences, all measured on this repo:

1. **Never read the statevector back to the CPU if a reduction can stay on GPU.**
   `metalq_expectation` (per-term partial sums only) vs readback+CPU contraction:
   3.9–6.1× faster at 20–24 qubits (357 ms vs 1401 ms at 24q).
2. **Fuse Pauli-term operations into one pass.** ⟨i|P|j⟩ is nonzero only for
   j = i⊕x_mask with element g·(−1)^popcount(i & (y|z)), g = (−i)^nY a per-term
   constant. So ⟨ψ|P|ψ⟩, ⟨λ|P|ψ⟩ and λ += c·P|ψ⟩ are each ONE gather-pass
   kernel (`pauli_expectation`/`pauli_inner_product`/`pauli_accumulate`) —
   no scratch copy, no blit, no per-qubit gate dispatches. Replacing the old
   copy+gates+reduce path (~6 passes/term) gave 1.5–1.9× on expectation and
   1.6–1.8× on adjoint gradients on top of the GPU-resident win.
3. **Fuse gates to cut dispatches.** `encode_circuit` multiplies runs of
   consecutive 1q gates on the same qubit into one 2×2 matrix on the CPU
   (`cmul2x2`) and dispatches once. Extend this pattern rather than bypassing it.
4. **Pool buffers on the context.** `mq_pool_acquire/release` (state_vector.m)
   reuses statevector/partials buffers across calls (≤4 idle per size);
   re-init is GPU-side (`encode_statevector_reset` = CPU-write of amp 0 +
   blit fill, `encode_zero_fill`), never a full CPU memset. Pool invariant:
   only release GPU-idle buffers (always waitUntilCompleted first).
5. **Batch command buffers.** Adjoint diff encodes forward pass + co-state +
   backward pass into as few command buffers as possible and only syncs when the
   partials buffer fills. Result: ≥25× faster than parameter-shift at 40 params.
6. Per-dispatch overhead dominates below ~16 qubits; don't chase GPU wins there.

## Kernel authoring rules (.metal)

- `struct Complex { float real; float imag; }` must stay byte-identical across
  every `.metal` file and `mq_complex_t` in metalq.h. Use the value-copying free
  functions (`c_mul`/`z_mul`, …) — they sidestep address-space qualifier issues.
- **k-qubit gate indexing**: each thread owns one group of 2^k amplitudes.
  Compute the base index by inserting a 0-bit at each target position in
  ascending order: `base = (base / step) * (step << 1) + (base % step)`.
  Thread count is `2^n / 2^k`, NOT 2^n.
- **Row-index convention**: `qubits[0]` is the MSB of the matrix row index
  (2q: row = control·2 + target; 3q: row = q0·4 + q1·2 + q2). Host-side
  `populate_matrix_*` and kernels must agree — this was a real CP-gate bug.
- **Tree reductions need a `threadgroup_barrier` at EVERY step.** The
  barrier-less "warp-synchronous" tail (`s < 32`) is NOT valid on Apple GPUs —
  the compiler may cache threadgroup memory in registers. See
  `pauli_expectation` for the canonical 256-wide reduction.
- Small per-gate constants (matrices, qubit indices) go via `setBytes`, not
  dedicated buffers.

## Dispatch rules (host side)

- Plain element-wise kernels: `dispatchThreads` (non-uniform threadgroups OK,
  supported on all Apple Silicon).
- Reduction kernels: `dispatchThreadgroups` with the exact group size the kernel
  hardcodes (256). All threads of a group must be live; out-of-range threads
  contribute zero via the `id < total_elements` guard. Using `dispatchThreads`
  here silently truncates the last group and corrupts the reduction.
- Pipelines are cached per kernel-name in the context's NSMutableDictionary via
  `get_pipeline(ctx, @"name")`. Never create a PSO inline.

## Command-buffer ordering (a real race)

Within ONE command buffer, Metal's hazard tracking orders compute and blit
encoders that touch the same buffer, and the default serial dispatch type
orders dispatches WITHIN a compute encoder (`encode_circuit` and the
read-modify-write `pauli_accumulate` chain rely on this — don't switch to
`MTLDispatchTypeConcurrent` without adding memory barriers). **Across command
buffers on the same queue there is NO such guarantee — they may overlap
execution.** The expectation
kernel encodes the forward pass into the FIRST batch's command buffer for
exactly this reason. When splitting work into batches, dependent work must
either live in the same command buffer or be separated by `waitUntilCompleted`.

Always check `cmd.error` after `waitUntilCompleted` and return a nonzero code.

## Bounded partials-buffer pattern (reductions with many outputs)

For N reductions (per Pauli term, per parameter), don't allocate N full buffers:

1. One partials buffer, capped (`256 MB`), sliced per reduction.
2. Slice stride = threadgroup count × sizeof(mq_complex_t), **rounded up to
   256 bytes** — `setBuffer:offset:` requires 256-byte alignment.
3. When slices run out: commit, `waitUntilCompleted`, CPU-sum each slice,
   start a new command buffer (flush pattern in `metalq_gradient_adjoint`).

## Precision strategy (measured, counterintuitive)

Statevector is float32 (mq_complex_t). The accuracy killer is not gate
application — it is **flat float32 accumulation over 2^n elements**:

- Readback + `np.vdot` on complex64 at 24 qubits: error ≈ 3.8e-3.
- GPU 256-wide float32 tree reduction, then CPU sums the ~65k per-group
  partials **in double**: error ≈ 1e-7 against a complex128 reference.

Rules: reduce hierarchically on GPU; accumulate anything that crosses to the
CPU in `double`; compute host-side matrix elements in double before narrowing
to float (`populate_matrix_*` takes `double *params`).

## C API / ctypes bridge rules

- Struct layouts in `metalq/backends/mps/backend.py` (MQComplex, MQGate,
  MQHamiltonian) must match `metalq.h` field-for-field. `mq_gate_t` is fixed
  arrays (`qubits[3]`, `params[3]`), not pointers.
- `mq_hamiltonian_t` holds raw pointers (`coeffs`, `pauli_codes`); the Python
  marshaler must return a keepalive holding the backing arrays, or the GPU
  reads freed memory.
- Pointer-returning functions (`metalq_create_context`) and bool-returning ones
  (`metalq_is_supported`) must be EXCLUDED from generic int-return error
  mapping — a context pointer interpreted as an error code was a real bug.
  Keep the exclusion lists in `backend.py` and `native_wrapper.py` in sync.
- New exported functions: declare in `metalq.h`, feature-detect in Python with
  `hasattr(self._lib, 'name')` and keep a fallback path so old dylibs still work.
- ObjC context lifecycle uses `__bridge_retained` + `CFRelease`; keep the
  pattern when adding fields to MetalQContext (and free them in destroy).
- **Every exported C function wraps its whole body in `@autoreleasepool`** —
  ctypes calls arrive on Python worker threads that have no pool; without one,
  autoreleased Metal objects accumulate for the thread's lifetime.
- Shared mutable context state must be locked: the pipeline cache and buffer
  pool are `@synchronized` — concurrent native calls on one context are legal
  (raw ctypes bypasses the Python-side `_call_with_timeout` lock).
- Encoding failure = fail closed: `populate_matrix_*` return bool,
  `encode_gate`/`encode_circuit` return int, exported functions map that to
  error code −5 (never apply identity and continue). Unmappable gate names
  raise `ValidationError` in `_marshal_gates` — no type-999 fallthrough.

## Adding a new GPU operation — checklist

1. Kernel in an existing `.metal` file (or new file + Makefile edit).
2. Host function in a new/existing `.m`: buffers via
   `create_statevector_buffer`, encode with `encode_circuit`/`encode_gate`,
   reduce with `complex_inner_product`, batch command buffers.
3. Export in `metalq.h`; wire ctypes signatures in `backend.py` (+ timeout via
   `_call_with_timeout`, scaled by 2^n and gate count).
4. Feature-detect with `hasattr` + fallback.
5. Correctness test in `tests/test_gpu_correctness.py` against the CPU backend.
6. Benchmark ≥3 sizes (e.g. 16/20/24q) before claiming a speedup; report the
   min of repeated runs after a warmup call.
