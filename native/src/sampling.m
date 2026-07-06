/*
 * sampling.m - GPU-resident measurement sampling
 *
 * Draws `shots` basis-state samples from |psi> = U|0...0> WITHOUT reading the
 * full statevector back to Python. The circuit runs on GPU, then a per-block
 * |amp|^2 tree-reduction (block_probability_sums) produces a coarse
 * probability histogram over 256-element blocks. A shot is then resolved
 * two-level entirely CPU-side with no further GPU work:
 *   1. binary-search a double-precision inclusive CDF over the block sums, then
 *   2. walk the chosen 256-element block by reading |amp|^2 directly from the
 *      SHARED (unified-memory) statevector buffer — only that block's ~2 KB
 *      page is touched.
 * Per-shot cost is O(log #blocks + 256) with a deterministic seeded PRNG, so
 * identical seeds give identical samples.
 */

#import "context_internal.h"
#import "metalq.h"
#import <Metal/Metal.h>

// Deterministic PRNG: splitmix64. Same seed -> same stream, independent of the
// platform C library (do NOT use arc4random/rand — those are not reproducible).
static inline uint64_t splitmix64(uint64_t *state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

int metalq_sample(mq_context_t ctx, uint32_t num_qubits, const mq_gate_t *gates,
                  uint32_t num_gates, uint32_t shots, uint64_t seed,
                  uint32_t *out_samples) {
  @autoreleasepool {
    if (!ctx || !out_samples)
      return -1;
    if (num_gates > 0 && !gates)
      return -1;
    if (shots == 0)
      return 0;

    MetalQContext *mCtx = (MetalQContext *)ctx;
    id<MTLDevice> device = (__bridge id<MTLDevice>)mCtx->device;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mCtx->commandQueue;
    if (!device || !queue)
      return -1;

    id<MTLComputePipelineState> reduce_pso =
        get_pipeline(mCtx, @"block_probability_sums");
    if (!reduce_pso) {
      NSLog(@"[MetalQ] Error: block_probability_sums pipeline not found");
      return -2;
    }

    // 1. Buffers (pooled)
    uint64_t num_elements = 1ULL << num_qubits;
    uint64_t sv_bytes = num_elements * sizeof(mq_complex_t);
    id<MTLBuffer> psi = mq_pool_acquire(mCtx, device, sv_bytes);
    if (!psi)
      return -3;

    uint64_t block_size = 256;
    uint64_t grid_size = (num_elements + block_size - 1) / block_size;
    uint64_t block_sums_bytes = grid_size * sizeof(float);

    id<MTLBuffer> block_sums = mq_pool_acquire(mCtx, device, block_sums_bytes);
    if (!block_sums) {
      mq_pool_release(mCtx, psi);
      return -3;
    }

    // Forward pass |psi> = U|0...0> then the block reduction, in ONE command
    // buffer so intra-command-buffer hazard tracking orders the reduction after
    // the circuit (separate command buffers on one queue may overlap).
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    encode_statevector_reset(cmd, psi, num_qubits);

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    int enc_rc = encode_circuit(enc, mCtx, num_qubits, gates, num_gates, psi);
    if (enc_rc != 0) {
      [enc endEncoding];
      mq_pool_release(mCtx, block_sums);
      mq_pool_release(mCtx, psi);
      return -5;
    }

    uint32_t total_elements = (uint32_t)num_elements;
    [enc setComputePipelineState:reduce_pso];
    [enc setBuffer:psi offset:0 atIndex:0];
    [enc setBuffer:block_sums offset:0 atIndex:1];
    [enc setBytes:&total_elements length:sizeof(total_elements) atIndex:2];
    // Uniform threadgroups: the reduction kernel requires all 256 threads of a
    // group to be live (out-of-range threads contribute zero).
    [enc dispatchThreadgroups:MTLSizeMake(grid_size, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(block_size, 1, 1)];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];

    if (cmd.error) {
      NSLog(@"[MetalQ] Sampling command buffer error: %@", cmd.error);
      mq_pool_release(mCtx, block_sums);
      mq_pool_release(mCtx, psi);
      return -4;
    }

    // 2. Build a double-precision inclusive CDF over the block sums. The total is
    // slightly != 1.0 (float32 gate accumulation over 2^n elements), so shots are
    // drawn against the ACTUAL total, never an assumed 1.0.
    const float *bs = (const float *)block_sums.contents;
    double *cdf = (double *)malloc(grid_size * sizeof(double));
    if (!cdf) {
      mq_pool_release(mCtx, block_sums);
      mq_pool_release(mCtx, psi);
      return -3;
    }
    double running = 0.0;
    for (uint64_t g = 0; g < grid_size; g++) {
      running += (double)bs[g];
      cdf[g] = running;
    }
    double total = running;

    const mq_complex_t *amps = (const mq_complex_t *)psi.contents;
    uint64_t state = seed;

    if (total <= 0.0) {
      // Degenerate (all-zero) state: nothing to sample from.
      for (uint32_t s = 0; s < shots; s++)
        out_samples[s] = 0;
      free(cdf);
      mq_pool_release(mCtx, block_sums);
      mq_pool_release(mCtx, psi);
      return 0;
    }

    // 3. For each shot, inverse-CDF sample: draw u in [0, total).
    for (uint32_t s = 0; s < shots; s++) {
      // 53-bit mantissa uniform in [0,1) scaled to [0, total).
      double unit =
          (double)(splitmix64(&state) >> 11) * (1.0 / 9007199254740992.0);
      double u = unit * total;

      // Binary search: smallest block b with cdf[b] > u. `>` skips zero-probability
      // blocks (their cdf equals the previous block's).
      uint64_t lo = 0, hi = grid_size - 1;
      while (lo < hi) {
        uint64_t mid = (lo + hi) >> 1;
        if (cdf[mid] > u)
          hi = mid;
        else
          lo = mid + 1;
      }
      uint64_t b = lo;
      double prev = (b > 0) ? cdf[b - 1] : 0.0;
      double r = u - prev; // remaining probability within the chosen block

      // Resolve within the 256-element block by reading |amp|^2 directly from the
      // shared statevector buffer. Track the last nonzero-probability element to
      // clamp against float round-off if the running sum never reaches r.
      uint64_t base = b * block_size;
      uint64_t end = base + block_size;
      if (end > num_elements)
        end = num_elements;

      double run = 0.0;
      uint64_t chosen = base;
      int64_t last_nonzero = -1;
      bool found = false;
      for (uint64_t i = base; i < end; i++) {
        double re = (double)amps[i].real;
        double im = (double)amps[i].imag;
        double pr = re * re + im * im;
        if (pr > 0.0)
          last_nonzero = (int64_t)i;
        run += pr;
        if (run > r) {
          chosen = i;
          found = true;
          break;
        }
      }
      if (!found)
        chosen = (last_nonzero >= 0) ? (uint64_t)last_nonzero : (end - 1);

      out_samples[s] = (uint32_t)chosen;
    }

    free(cdf);
    mq_pool_release(mCtx, block_sums);
    mq_pool_release(mCtx, psi);
    return 0;
  }
}
