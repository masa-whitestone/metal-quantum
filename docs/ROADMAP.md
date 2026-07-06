# Roadmap — Future Improvements

Backlog of known gaps and optimization opportunities in the GPU (MPS) backend,
identified during the 2026-07 native-layer audit. Items are ordered by
expected impact within each section. Baseline numbers below are from an
Apple M3 Pro, float32 statevector.

## Performance

### QFT / diagonal-gate fusion (highest impact, known loss)
A benchmark against osxQ (MLX-based simulator) showed Metal-Q **~2× slower on
QFT at 24 qubits** (625 ms vs 317 ms) while winning every other workload
(circuit execution 7–16×, expectation 2.6–4.9×). The cause: Metal-Q executes
the QFT phase ladder as individual CP gate dispatches — n(n−1)/2 + n ≈ 300
full statevector passes at 24 qubits — while osxQ fuses the ladder into
diagonal stages. Fix: detect runs of consecutive *diagonal* gates (CP, CZ, P,
RZ, CCZ) in `encode_circuit` and fuse them into a single pass with a
per-element phase kernel (phases computable in-kernel from the gate list, or
via a precomputed per-index phase accumulation). This generalizes the existing
1q-gate fusion and should flip the QFT result.

### Multi-term batching for diagonal (Z-heavy) Hamiltonians
`metalq_expectation` runs one fused-kernel pass per Pauli term. All diagonal
terms (Z-strings — the majority in Ising-type Hamiltonians) read the same
|ψ|² values; k diagonal terms could share ONE pass that accumulates k signed
partial sums (one threadgroup array per term, or a term-loop inside the
kernel). Expected: expectation cost for Ising Hamiltonians approaches a
single statevector pass regardless of term count.

### Persistent statevector handle (VQE-scale win)
Every API call (`run`/`expectation`/`gradient`/`sample`) re-executes the
circuit from |0…0⟩. A context-owned persistent state handle
(`metalq_state_create/apply/expectation/sample/destroy`) would let a VQE loop
run the ansatz once per iteration and query multiple observables, or let
users compose circuits incrementally. `expectation_and_gradient` already
fuses the two most common calls; this generalizes it.

### GPU sampling: within-block resolution cost
`metalq_sample` resolves each shot CPU-side with an O(256) walk inside the
chosen block (~450 ms CPU for 10⁶ shots at 24q). Options: vectorize the
within-block CDF with NEON/vDSP, sort the random draws for cache locality, or
build a GPU alias table for very high shot counts.

### Minor
- Second-stage GPU reduction for partial sums (currently CPU-summed in
  double; correct but serial — only matters above ~26 qubits).
- Threadgroup size is hardcoded to 256; could tune per device via
  `maxTotalThreadsPerThreadgroup` / `threadExecutionWidth`.
- Backward pass in adjoint diff calls `encode_gate` per inverse gate without
  the 1q-fusion used in forward passes.

## Features

### Arbitrary unitary gate GPU path
`mq_gate_t` has no matrix payload, so `UnitaryGate`/`DiagonalGate`-style
custom matrices cannot reach the GPU (they fail closed with
`ValidationError`). The kernels (`apply_gate_single/two/three`) already take
arbitrary matrices — only the C API and marshaling need a variant that
carries an inline 2×2/4×4/8×8 matrix. This also unblocks the Qiskit adapter
for circuits containing unitary instructions.

### Remaining named gates
Fail-closed today (raise `ValidationError` on MPS): `crx/cry/crz`, `cu/cu1/cu3`,
`csx`, `iswap`, `dcx`, `ecr`, `rxx/ryy/rzz/rzx`. The controlled rotations and
two-qubit rotations are straightforward `populate_matrix_two` entries; the
Pauli-rotation pairs (rxx/ryy/rzz) could alternatively reuse the fused
Pauli-kernel machinery (they are exp(−iθ/2·P⊗P)).

### Native adjoint gradient for U3
Parameterized `u3` falls back to parameter-shift for the whole circuit. The
three U3 generators can be expressed as weighted Pauli-slice sums (same
mechanism as the P/CP gradients in `generator_slices`), keeping VQE ansätze
with u3 layers on the fast adjoint path.

### GPU sampling for `sx` global phase
`sx` marshals to RX(π/2), equal to SX only up to global phase e^{iπ/4}.
Unobservable in probabilities/expectations, but the statevector differs from
Qiskit's. Add a dedicated matrix entry if exact statevector parity matters.

## Robustness / infrastructure

- **Legacy test suite**: 70 tests (test_all_gates, test_high_level_gates,
  test_basic, test_rotation_gates, test_api_basic, test_circuit_extended,
  test_phase1) predate the Circuit-builder API rewrite and pass Qiskit
  `QuantumCircuit` objects directly. Either auto-convert Qiskit circuits in
  `api.py` via `adapters/qiskit_adapter.py`, or rewrite the tests.
- **Duplicate secure backends**: `secure_backend.py` and
  `secure_mps_backend.py` overlap; consolidate to one module (both are
  currently referenced by different test files).
- **Packaging**: the backend loads `native/build/libmetalq.dylib`; the stale
  `metalq/lib/` artifacts and the wheel-packaging story (bundling the dylib +
  metallib) need reconciling before distribution.
- `native_wrapper.py` (secure ctypes wrapper) lags `backend.py`'s feature set
  (no `metalq_expectation`/`metalq_sample`/`metalq_gradient_adjoint_energy`).
- Early-return error paths in `metalq_gradient_adjoint_energy` drop pooled
  buffers back to ARC instead of the pool (correct, but pool misses after
  errors).

## Reference benchmarks (2026-07-06, M3 Pro)

| Workload | 16q | 20q | 22q | 24q |
|---|---|---|---|---|
| Circuit execution (3-layer ansatz) | 1.5 ms | 6.7 ms | 65 ms | 288 ms |
| Expectation (2n−1 Pauli terms) | 2.2 ms | 10.3 ms | 82 ms | 367 ms |
| QFT | 2.1 ms | 11.3 ms | 132 ms | 625 ms |
| Adjoint gradient (2n params) | 2.6 ms (16q) | 32 ms (20q) | 180 ms (22q) | — |
| Fused E+∇E VQE iteration (20q/40p) | — | 34 ms | — | — |
| Sampling 10⁴ shots | — | — | — | 286 ms |

Accuracy: expectation matches complex128 reference to ~1e-7 at 24q; adjoint
gradients match parameter-shift to ~1e-7.
