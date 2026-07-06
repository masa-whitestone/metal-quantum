/*
 * adjoint_diff.m - Adjoint Differentiation Implementation
 *
 * Computes d<psi|H|psi>/dtheta for all circuit parameters in O(num_gates)
 * gate applications (vs O(num_gates * num_params) for parameter shift).
 *
 * The co-state build and per-parameter overlaps use the fused Pauli-string
 * kernels (one pass per term / per parameter, no scratch buffer, no blits).
 * The whole forward pass, co-state initialization and backward pass are
 * encoded into as few command buffers as possible: per-parameter inner
 * products land in distinct slices of one partial-sums buffer, and the CPU
 * only synchronizes when a batch of slices is full (or at the end).
 */

#import "context_internal.h"
#import "metalq.h"
#import <Metal/Metal.h>

// Build the inverse of a gate. Handles the non-self-inverse gates the
// simulator supports; self-inverse gates pass through unchanged.
static mq_gate_t invert_gate(mq_gate_t gate) {
  mq_gate_t inv = gate;
  switch (gate.type) {
  case MQ_GATE_RX:
  case MQ_GATE_RY:
  case MQ_GATE_RZ:
  case MQ_GATE_P:
  case MQ_GATE_U1:
  case MQ_GATE_CP:
    inv.params[0] = -gate.params[0];
    break;
  case MQ_GATE_S:
    inv.type = MQ_GATE_P;
    inv.params[0] = -M_PI / 2;
    inv.num_params = 1;
    break;
  case MQ_GATE_T:
    inv.type = MQ_GATE_P;
    inv.params[0] = -M_PI / 4;
    inv.num_params = 1;
    break;
  case MQ_GATE_U3:
    // U3(t,phi,lam)^{-1} = U3(-t, -lam, -phi)
    inv.params[0] = -gate.params[0];
    inv.params[1] = -gate.params[2];
    inv.params[2] = -gate.params[1];
    break;
  default:
    break; // X, Y, Z, H, CX, CY, CZ, SWAP, CH, CCX, CSWAP, CCZ are self-inverse
  }
  return inv;
}

// One Pauli-string reduction plus a CPU-side weight. Each parameterized gate
// contributes 1..4 of these; out_gradients[param] += sum(weight * Im<lambda|P|psi>).
typedef struct {
  mq_pauli_params_t masks;
  double weight;
} mq_grad_slice_t;

// Fill the reduction slices for a parameterized gate. Returns the number of
// slices (0 = gradient for this gate type is not implemented natively; the
// Python layer falls back to parameter-shift before reaching this point).
//
// Convention (matches the RX case the rest of the code assumes): for a gate
// U(theta) with dU/dtheta * U^{-1} = A, grad = 2 Re<lambda|A|psi>. For a Pauli
// generator P (U = exp(-i theta P / 2)) this is Im<lambda|P|psi>.
//   RX: A = -i/2 X            -> grad =  Im<lambda|X|psi>
//   RY: A = -i/2 Y            -> grad =  Im<lambda|Y|psi>
//   RZ: A = -i/2 Z            -> grad =  Im<lambda|Z|psi>
//   P(theta)=diag(1,e^{i t}) = exp(i t nhat), nhat=(I-Z)/2, dU/dt U^-1 = i nhat
//     grad = 2 Re<lambda|i nhat|psi> = Im<lambda|Z_q|psi> - Im<lambda|psi>
//   CP(theta): generator |11><11| = (I - Z_c - Z_t + Z_c Z_t)/4
//     grad = -1/2 [Im<lambda|psi> - Im<lambda|Z_c|psi>
//                  - Im<lambda|Z_t|psi> + Im<lambda|Z_c Z_t|psi>]
static int generator_slices(const mq_gate_t *gate, uint32_t num_elements,
                            mq_grad_slice_t *out) {
  // Helper to initialise an identity (all-zero-mask) reduction.
#define INIT_MASKS(P)                                                          \
  do {                                                                         \
    (P)->x_mask = 0;                                                           \
    (P)->y_mask = 0;                                                           \
    (P)->z_mask = 0;                                                           \
    (P)->total_elements = num_elements;                                        \
    (P)->g_re = 1.0f;                                                          \
    (P)->g_im = 0.0f;                                                          \
  } while (0)

  uint32_t bit = 1u << gate->qubits[0];

  if (gate->type == MQ_GATE_RX) {
    INIT_MASKS(&out[0].masks);
    out[0].masks.x_mask = bit;
    out[0].weight = 1.0;
    return 1;
  } else if (gate->type == MQ_GATE_RY) {
    INIT_MASKS(&out[0].masks);
    out[0].masks.x_mask = bit;
    out[0].masks.y_mask = bit;
    out[0].masks.g_re = 0.0f;
    out[0].masks.g_im = -1.0f; // (-i)^1
    out[0].weight = 1.0;
    return 1;
  } else if (gate->type == MQ_GATE_RZ) {
    INIT_MASKS(&out[0].masks);
    out[0].masks.z_mask = bit;
    out[0].weight = 1.0;
    return 1;
  } else if (gate->type == MQ_GATE_P || gate->type == MQ_GATE_U1) {
    // Im<lambda|Z_q|psi> - Im<lambda|psi>
    INIT_MASKS(&out[0].masks);
    out[0].masks.z_mask = bit;
    out[0].weight = 1.0;
    INIT_MASKS(&out[1].masks); // identity
    out[1].weight = -1.0;
    return 2;
  } else if (gate->type == MQ_GATE_CP) {
    uint32_t bit_c = 1u << gate->qubits[0];
    uint32_t bit_t = 1u << gate->qubits[1];
    INIT_MASKS(&out[0].masks); // identity
    out[0].weight = -0.5;
    INIT_MASKS(&out[1].masks); // Z_c
    out[1].masks.z_mask = bit_c;
    out[1].weight = 0.5;
    INIT_MASKS(&out[2].masks); // Z_t
    out[2].masks.z_mask = bit_t;
    out[2].weight = 0.5;
    INIT_MASKS(&out[3].masks); // Z_c Z_t
    out[3].masks.z_mask = bit_c | bit_t;
    out[3].weight = -0.5;
    return 4;
  }
  return 0;
#undef INIT_MASKS
}

// Cap for the partial-sums buffer (per-parameter slices).
#define MQ_ADJOINT_PARTIALS_CAP (256ULL * 1024 * 1024)

int metalq_gradient_adjoint(mq_context_t ctx, uint32_t num_qubits,
                            const mq_gate_t *gates, uint32_t num_gates,
                            void *hamiltonian, double *out_gradients) {
  @autoreleasepool {
    return metalq_gradient_adjoint_energy(ctx, num_qubits, gates, num_gates,
                                          hamiltonian, out_gradients, NULL);
  }
}

int metalq_gradient_adjoint_energy(mq_context_t ctx, uint32_t num_qubits,
                                   const mq_gate_t *gates, uint32_t num_gates,
                                   void *hamiltonian, double *out_gradients,
                                   double *out_energy) {
  @autoreleasepool {
    if (!ctx || !gates || !hamiltonian || !out_gradients)
      return -1;

    MetalQContext *mCtx = (MetalQContext *)ctx;
    id<MTLDevice> device = (__bridge id<MTLDevice>)mCtx->device;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mCtx->commandQueue;

    mq_hamiltonian_t *H = (mq_hamiltonian_t *)hamiltonian;

    id<MTLComputePipelineState> reduce_pso =
        get_pipeline(mCtx, @"pauli_inner_product");
    id<MTLComputePipelineState> accum_pso =
        get_pipeline(mCtx, @"pauli_accumulate");
    if (!reduce_pso || !accum_pso) {
      NSLog(@"[MetalQ] Error: fused Pauli pipelines not found");
      return -2;
    }

    uint32_t total_param_count = 0;
    for (uint32_t i = 0; i < num_gates; i++)
      total_param_count += gates[i].num_params;

    // Gradients accumulate via += across (possibly multiple) reduction slices per
    // parameter and across batch flushes, so start from zero. Params whose gate
    // type has no native generator (e.g. constant U3) keep their 0 entry.
    for (uint32_t i = 0; i < total_param_count; i++)
      out_gradients[i] = 0.0;

    // 1. Buffers (pooled). The fused kernels remove the scratch buffer.
    uint64_t num_elements = 1ULL << num_qubits;
    uint64_t sv_bytes = num_elements * sizeof(mq_complex_t);
    id<MTLBuffer> psi = mq_pool_acquire(mCtx, device, sv_bytes);
    id<MTLBuffer> lambda = mq_pool_acquire(mCtx, device, sv_bytes);
    if (!psi || !lambda)
      return -3;

    uint64_t block_size = 256;
    uint64_t grid_size = (num_elements + block_size - 1) / block_size;

    // Per-parameter slice of the partials buffer, 256-byte aligned for
    // setBuffer:offset:.
    uint64_t slice_bytes =
        ((grid_size * sizeof(mq_complex_t) + 255ULL) / 256ULL) * 256ULL;
    uint32_t slices_per_batch =
        (uint32_t)(MQ_ADJOINT_PARTIALS_CAP / slice_bytes);
    if (slices_per_batch == 0)
      slices_per_batch = 1;
    if (total_param_count > 0 && slices_per_batch > total_param_count)
      slices_per_batch = total_param_count;

    id<MTLBuffer> partials = nil;
    uint32_t *slice_param = NULL;  // slice index -> flat parameter index
    double *slice_weight = NULL;   // slice index -> CPU-side weight
    if (total_param_count > 0) {
      partials = mq_pool_acquire(mCtx, device, slice_bytes * slices_per_batch);
      slice_param = malloc(slices_per_batch * sizeof(uint32_t));
      slice_weight = malloc(slices_per_batch * sizeof(double));
      if (!partials || !slice_param || !slice_weight) {
        free(slice_param);
        free(slice_weight);
        return -3;
      }
    }

    // Dedicated small partials buffer for the fused energy reduction (identity
    // Pauli dispatch). Kept separate from the per-parameter slice bookkeeping
    // above so it doesn't disturb it.
    uint64_t energy_slice_bytes =
        ((grid_size * sizeof(mq_complex_t) + 255ULL) / 256ULL) * 256ULL;
    id<MTLBuffer> energy_partials = nil;
    bool energy_read = false;
    if (out_energy) {
      energy_partials = mq_pool_acquire(mCtx, device, energy_slice_bytes);
      if (!energy_partials) {
        free(slice_param);
        free(slice_weight);
        return -3;
      }
    }

    uint32_t total_elems_u32 = (uint32_t)num_elements;

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc;

    // 2. GPU-side init: psi = |0...0>, lambda = 0 (accumulator)
    encode_statevector_reset(cmd, psi, num_qubits);
    encode_zero_fill(cmd, lambda, sv_bytes);

    // 3. Forward Pass: |psi> = U|0...0>
    enc = [cmd computeCommandEncoder];
    if (encode_circuit(enc, mCtx, num_qubits, gates, num_gates, psi) != 0) {
      [enc endEncoding];
      free(slice_param);
      free(slice_weight);
      mq_pool_release(mCtx, partials);
      mq_pool_release(mCtx, energy_partials);
      mq_pool_release(mCtx, lambda);
      mq_pool_release(mCtx, psi);
      return -5;
    }

    // 4. Co-state |lambda> = H|psi>: one fused dispatch per term (the serial
    // encoder orders the read-modify-writes on lambda).
    for (uint32_t j = 0; j < H->num_terms; j++) {
      mq_pauli_params_t p;
      p.total_elements = total_elems_u32;
      mq_term_masks(H, j, &p.x_mask, &p.y_mask, &p.z_mask, &p.g_re, &p.g_im);
      float coeff = (float)H->coeffs[j];
      p.g_re *= coeff;
      p.g_im *= coeff;

      [enc setComputePipelineState:accum_pso];
      [enc setBuffer:psi offset:0 atIndex:0];
      [enc setBuffer:lambda offset:0 atIndex:1];
      [enc setBytes:&p length:sizeof(p) atIndex:2];

      uint64_t tpg = (num_elements < 256) ? num_elements : 256;
      [enc dispatchThreads:MTLSizeMake(num_elements, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    }

    // 4b. Energy: E = Re<psi|lambda>, one identity-mask pauli_inner_product
    // dispatch in the SAME encoder (so it is serially ordered after the co-state
    // build above and before the backward pass below, which uncomputes psi in
    // place). bra=psi, ket=lambda, identity masks (x=y=z=0, g=1+0i) => term =
    // conj(psi_i) * lambda_i, so the real-part sum is Re<psi|lambda>.
    if (out_energy) {
      mq_pauli_params_t ip;
      ip.x_mask = 0;
      ip.y_mask = 0;
      ip.z_mask = 0;
      ip.total_elements = total_elems_u32;
      ip.g_re = 1.0f;
      ip.g_im = 0.0f;

      [enc setComputePipelineState:reduce_pso];
      [enc setBuffer:psi offset:0 atIndex:0];
      [enc setBuffer:lambda offset:0 atIndex:1];
      [enc setBuffer:energy_partials offset:0 atIndex:2];
      [enc setBytes:&ip length:sizeof(ip) atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake(grid_size, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(block_size, 1, 1)];
    }
    [enc endEncoding];

    // 5. Backward Pass. Everything stays in the current command buffer; the
    // CPU only synchronizes when the partials buffer is full.
    uint32_t current_param_index = total_param_count;
    uint32_t batch_used = 0;

    for (int i = (int)num_gates - 1; i >= 0; i--) {
      mq_gate_t gate = gates[i];
      current_param_index -= gate.num_params;
      mq_gate_t inv_gate = invert_gate(gate);

      // a. Gradient for parameterized gates. Each parameter contributes 1..4
      // reduction slices, each with a CPU-side weight; they accumulate via +=
      // into out_gradients[param] (see generator_slices).
      for (uint32_t p_idx = 0; p_idx < gate.num_params; p_idx++) {
        mq_grad_slice_t slices[4];
        int nslices = generator_slices(&gate, total_elems_u32, slices);
        if (nslices == 0) {
          // Gradient for this gate type is not implemented natively (e.g. U3);
          // the Python layer falls back to parameter-shift before reaching here,
          // so the output entry simply stays 0.
          continue;
        }
        uint32_t param_index = current_param_index + p_idx;

        for (int sl = 0; sl < nslices; sl++) {
          if (batch_used == slices_per_batch) {
            // Flush: run everything queued so far and read back the slices.
            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.error) {
              NSLog(@"[MetalQ] Adjoint command buffer error: %@", cmd.error);
              free(slice_param);
              free(slice_weight);
              return -4;
            }
            if (out_energy && !energy_read) {
              mq_complex_t *eps = (mq_complex_t *)energy_partials.contents;
              double e = 0.0;
              for (uint64_t k = 0; k < grid_size; k++)
                e += eps[k].real;
              *out_energy = e;
              energy_read = true;
            }
            for (uint32_t s = 0; s < batch_used; s++) {
              mq_complex_t *ps =
                  (mq_complex_t *)((char *)partials.contents + s * slice_bytes);
              double dot_imag = 0.0;
              for (uint64_t k = 0; k < grid_size; k++)
                dot_imag += ps[k].imag;
              out_gradients[slice_param[s]] += slice_weight[s] * dot_imag;
            }
            batch_used = 0;
            cmd = [queue commandBuffer];
          }

          // Fused <lambda| P |psi> partial sums — no scratch copy, no blit.
          enc = [cmd computeCommandEncoder];
          [enc setComputePipelineState:reduce_pso];
          [enc setBuffer:lambda offset:0 atIndex:0];
          [enc setBuffer:psi offset:0 atIndex:1];
          [enc setBuffer:partials offset:batch_used * slice_bytes atIndex:2];
          [enc setBytes:&slices[sl].masks length:sizeof(slices[sl].masks)
                atIndex:3];
          // Uniform threadgroups: the reduction kernel requires all 256 threads
          // of a group to be live (out-of-range threads contribute zero).
          [enc dispatchThreadgroups:MTLSizeMake(grid_size, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(block_size, 1, 1)];
          [enc endEncoding];

          slice_param[batch_used] = param_index;
          slice_weight[batch_used] = slices[sl].weight;
          batch_used++;
        }
      }

      // b. Uncompute psi and backprop lambda
      enc = [cmd computeCommandEncoder];
      int rc_psi = encode_gate(enc, mCtx, num_qubits, &inv_gate, psi);
      int rc_lam = encode_gate(enc, mCtx, num_qubits, &inv_gate, lambda);
      [enc endEncoding];
      if (rc_psi != 0 || rc_lam != 0) {
        [cmd commit];
        [cmd waitUntilCompleted];
        free(slice_param);
        free(slice_weight);
        mq_pool_release(mCtx, partials);
        mq_pool_release(mCtx, energy_partials);
        mq_pool_release(mCtx, lambda);
        mq_pool_release(mCtx, psi);
        return -5;
      }
    }

    // 6. Final flush
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.error) {
      NSLog(@"[MetalQ] Adjoint command buffer error: %@", cmd.error);
      free(slice_param);
      free(slice_weight);
      return -4;
    }
    if (out_energy && !energy_read) {
      mq_complex_t *eps = (mq_complex_t *)energy_partials.contents;
      double e = 0.0;
      for (uint64_t k = 0; k < grid_size; k++)
        e += eps[k].real;
      *out_energy = e;
      energy_read = true;
    }
    for (uint32_t s = 0; s < batch_used; s++) {
      mq_complex_t *ps =
          (mq_complex_t *)((char *)partials.contents + s * slice_bytes);
      double dot_imag = 0.0;
      for (uint64_t k = 0; k < grid_size; k++)
        dot_imag += ps[k].imag;
      out_gradients[slice_param[s]] += slice_weight[s] * dot_imag;
    }

    free(slice_param);
    free(slice_weight);
    mq_pool_release(mCtx, partials);
    mq_pool_release(mCtx, energy_partials);
    mq_pool_release(mCtx, lambda);
    mq_pool_release(mCtx, psi);
    return 0;
  }
}
