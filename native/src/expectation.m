/*
 * expectation.m - GPU Expectation Value <psi|H|psi>
 *
 * Runs the circuit once on GPU, then evaluates every Pauli term with the
 * fused pauli_expectation kernel: <i|P|j> is nonzero only for j = i^x_mask,
 * so each term is ONE reduction pass over |psi> — no scratch copy, no blit,
 * no per-qubit gate dispatches. Only small per-term partial sums are read
 * back to the CPU; the full statevector never leaves the GPU.
 */

#import "context_internal.h"
#import "metalq.h"
#import <Metal/Metal.h>

// Cap for the per-batch partial-sums buffer. Terms are processed in batches
// so the buffer stays bounded for large qubit counts / many-term Hamiltonians.
#define MQ_EXPECTATION_PARTIALS_CAP (256ULL * 1024 * 1024)

int metalq_expectation(mq_context_t ctx, uint32_t num_qubits,
                       const mq_gate_t *gates, uint32_t num_gates,
                       void *hamiltonian, double *out_value) {
  @autoreleasepool {
    if (!ctx || !hamiltonian || !out_value)
      return -1;
    if (num_gates > 0 && !gates)
      return -1;

    MetalQContext *mCtx = (MetalQContext *)ctx;
    id<MTLDevice> device = (__bridge id<MTLDevice>)mCtx->device;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mCtx->commandQueue;
    if (!device || !queue)
      return -1;

    mq_hamiltonian_t *H = (mq_hamiltonian_t *)hamiltonian;

    id<MTLComputePipelineState> reduce_pso =
        get_pipeline(mCtx, @"pauli_expectation");
    if (!reduce_pso) {
      NSLog(@"[MetalQ] Error: pauli_expectation pipeline not found");
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

    // Per-term slice stride (float partials), 256-byte aligned for
    // setBuffer:offset:.
    uint64_t slice_bytes =
        ((grid_size * sizeof(float) + 255ULL) / 256ULL) * 256ULL;
    uint32_t terms_per_batch =
        (uint32_t)(MQ_EXPECTATION_PARTIALS_CAP / slice_bytes);
    if (terms_per_batch == 0)
      terms_per_batch = 1;
    if (terms_per_batch > H->num_terms)
      terms_per_batch = H->num_terms;

    id<MTLBuffer> partials =
        mq_pool_acquire(mCtx, device, slice_bytes * terms_per_batch);
    if (!partials)
      return -3;

    // 2+3. Forward pass |psi> = U|0...0>, then per-term reduction, batched.
    // The reset + forward pass are encoded into the FIRST batch's command
    // buffer so the reductions are ordered after the circuit by
    // intra-command-buffer hazard tracking (separate command buffers on one
    // queue may overlap execution).
    double expectation = 0.0;
    id<MTLCommandBuffer> cmd = nil;
    id<MTLComputeCommandEncoder> enc = nil;

    for (uint32_t batch_start = 0; batch_start < H->num_terms;
         batch_start += terms_per_batch) {
      uint32_t batch_end = batch_start + terms_per_batch;
      if (batch_end > H->num_terms)
        batch_end = H->num_terms;

      cmd = [queue commandBuffer];

      if (batch_start == 0) {
        encode_statevector_reset(cmd, psi, num_qubits);
        enc = [cmd computeCommandEncoder];
        int enc_rc = encode_circuit(enc, mCtx, num_qubits, gates, num_gates, psi);
        [enc endEncoding];
        if (enc_rc != 0) {
          mq_pool_release(mCtx, partials);
          mq_pool_release(mCtx, psi);
          return -5;
        }
      }

      enc = [cmd computeCommandEncoder];
      for (uint32_t j = batch_start; j < batch_end; j++) {
        mq_pauli_params_t p;
        p.total_elements = (uint32_t)num_elements;
        mq_term_masks(H, j, &p.x_mask, &p.y_mask, &p.z_mask, &p.g_re, &p.g_im);

        [enc setComputePipelineState:reduce_pso];
        [enc setBuffer:psi offset:0 atIndex:0];
        [enc setBuffer:partials
                offset:(j - batch_start) * slice_bytes
               atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];

        // Uniform threadgroups: the reduction kernel requires all 256 threads
        // of a group to be live (out-of-range threads contribute zero).
        [enc dispatchThreadgroups:MTLSizeMake(grid_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(block_size, 1, 1)];
      }
      [enc endEncoding];

      [cmd commit];
      [cmd waitUntilCompleted];

      if (cmd.error) {
        NSLog(@"[MetalQ] Expectation command buffer error: %@", cmd.error);
        return -4;
      }

      // 4. CPU-reduce the small per-term partials in double precision
      for (uint32_t j = batch_start; j < batch_end; j++) {
        float *ps =
            (float *)((char *)partials.contents + (j - batch_start) * slice_bytes);
        double dot = 0.0;
        for (uint64_t k = 0; k < grid_size; k++) {
          dot += ps[k];
        }
        expectation += H->coeffs[j] * dot;
      }
    }

    mq_pool_release(mCtx, partials);
    mq_pool_release(mCtx, psi);

    *out_value = expectation;
    return 0;
  }
}
