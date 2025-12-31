/*
 * adjoint_diff.m - Adjoint Differentiation Implementation
 */

#import "context_internal.h"
#import "metalq.h"
#import <Metal/Metal.h>

// Forward declaration of internal helpers from gate_executor.m
// We should probably move these to a shared internal header, but for now:
void encode_gate(id<MTLComputeCommandEncoder> encoder, MetalQContext *ctx,
                 uint32_t num_qubits, const mq_gate_t *gate,
                 id<MTLBuffer> stateBuffer);
id<MTLBuffer> create_statevector_buffer(id<MTLDevice> device,
                                        uint32_t num_qubits);
id<MTLComputePipelineState> get_pipeline(MetalQContext *ctx,
                                         NSString *kernelName);

// Helper to apply linear combination: dst += coeff * src
void apply_linear_combination(id<MTLComputeCommandEncoder> encoder,
                              MetalQContext *ctx, id<MTLBuffer> src,
                              id<MTLBuffer> dst, float coeff_real,
                              float coeff_imag, uint32_t num_qubits) {
  id<MTLComputePipelineState> pso =
      get_pipeline(ctx, @"complex_linear_combine");
  if (!pso)
    return;

  [encoder setComputePipelineState:pso];
  [encoder setBuffer:src offset:0 atIndex:0];
  [encoder setBuffer:dst offset:0 atIndex:1];
  [encoder setBytes:&coeff_real length:sizeof(float) atIndex:2];
  [encoder setBytes:&coeff_imag length:sizeof(float) atIndex:3];

  uint64_t num_elements = 1ULL << num_qubits;
  uint64_t tpg = (num_elements < 256) ? num_elements : 256;
  MTLSize gridSize = MTLSizeMake(num_elements, 1, 1);
  MTLSize threadsPerGroup = MTLSizeMake(tpg, 1, 1);
  [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];
}

// Helper to copy buffer: dst = src
void copy_buffer(id<MTLBlitCommandEncoder> encoder, id<MTLBuffer> src,
                 id<MTLBuffer> dst, uint32_t num_qubits) {
  uint64_t size = (1ULL << num_qubits) * sizeof(mq_complex_t);
  [encoder copyFromBuffer:src
             sourceOffset:0
                 toBuffer:dst
        destinationOffset:0
                     size:size];
}

int metalq_gradient_adjoint(mq_context_t ctx, uint32_t num_qubits,
                            const mq_gate_t *gates, uint32_t num_gates,
                            void *hamiltonian, double *out_gradients) {

  if (!ctx || !gates || !hamiltonian || !out_gradients)
    return -1;

  MetalQContext *mCtx = (MetalQContext *)ctx;
  id<MTLDevice> device = (__bridge id<MTLDevice>)mCtx->device;
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mCtx->commandQueue;

  mq_hamiltonian_t *H = (mq_hamiltonian_t *)hamiltonian;

  // 1. Buffers
  id<MTLBuffer> psi = create_statevector_buffer(device, num_qubits);
  id<MTLBuffer> lambda = create_statevector_buffer(device, num_qubits);
  id<MTLBuffer> temp = create_statevector_buffer(device, num_qubits);

  // Initialize lambda to 0 (create_statevector_buffer inits to |0>, need strict
  // 0 for accumulation) Actually create_statevector_buffer inits to |0>. We
  // need a "zeros" buffer. We can just memset lambda contents to 0 before use,
  // or run a clear kernel. Since storage is Shared, memset is fine.
  memset(lambda.contents, 0, lambda.length);

  // 2. Forward Pass (Compute |psi_final>)
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

  for (uint32_t i = 0; i < num_gates; i++) {
    encode_gate(enc, mCtx, num_qubits, &gates[i], psi);
  }
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted]; // Ensure psi is ready

  // 3. Initialize Co-State |lambda> = H |psi>
  // H = sum c_j P_j
  cmd = [queue commandBuffer];

  // We need to apply P_j to psi (into temp), then add c_j * temp to lambda.
  // Since we don't have a "reset temp" kernel, we rely on copy or just
  // overwrite. `encode_gate` reads/writes same buffer.

  for (uint32_t j = 0; j < H->num_terms; j++) {
    // a. Copy psi to temp
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    copy_buffer(blit, psi, temp, num_qubits);
    [blit endEncoding];

    // b. Apply Pauli product P_j to temp
    enc = [cmd computeCommandEncoder];

    // Construct gates from Pauli codes
    for (uint32_t q = 0; q < H->num_qubits; q++) {
      uint8_t code = H->pauli_codes[j * H->num_qubits + q];
      if (code == 0)
        continue; // Identity

      mq_gate_t p_gate;
      p_gate.num_qubits = 1;
      p_gate.qubits[0] = q;
      p_gate.num_params = 0;

      if (code == 1)
        p_gate.type = MQ_GATE_X;
      else if (code == 2)
        p_gate.type = MQ_GATE_Y;
      else if (code == 3)
        p_gate.type = MQ_GATE_Z;

      encode_gate(enc, mCtx, num_qubits, &p_gate, temp);
    }

    // c. Accumulate into lambda: lambda += c_j * temp
    double c_val = H->coeffs[j];
    apply_linear_combination(enc, mCtx, temp, lambda, (float)c_val, 0.0f,
                             num_qubits);

    [enc endEncoding];
  }
  [cmd commit];
  [cmd waitUntilCompleted];

  // 4. Backward Pass
  // first, then decrement index as we go backwards?

  uint32_t total_param_count = 0;
  for (uint32_t i = 0; i < num_gates; i++) {
    total_param_count += gates[i].num_params;
  }

  uint32_t current_param_index = total_param_count; // Points to end

  // Create Partial Sums Buffer
  // Max grid size? 2^N elements. Block size 256. Grid size 2^N / 256.
  uint64_t num_elements = 1ULL << num_qubits;
  uint64_t block_size = 256;
  uint64_t grid_size = (num_elements + block_size - 1) / block_size;
  id<MTLBuffer> partial_sums =
      [device newBufferWithLength:grid_size * sizeof(mq_complex_t)
                          options:MTLResourceStorageModeShared];

  id<MTLComputePipelineState> reduce_pso =
      get_pipeline(mCtx, @"complex_inner_product");
  if (!reduce_pso) {
    printf("Error: complex_inner_product pipeline not found\n");
    return -2;
  }

  // Iterate backwards
  for (int i = num_gates - 1; i >= 0; i--) {
    mq_gate_t gate = gates[i];

    // Decrement param index start for this gate
    current_param_index -= gate.num_params;

    // a. Prepare inverse gate
    mq_gate_t inv_gate = gate;
    if (gate.type >= MQ_GATE_RX && gate.type <= MQ_GATE_RZ) { // RX, RY, RZ
      inv_gate.params[0] = -gate.params[0];
    }

    // b. Compute Gradient if parameterized
    // Currently supports 1 param gates (RX, RY, RZ).
    // Loop over params if needed (future proofing).
    for (int p_idx = 0; p_idx < gate.num_params; p_idx++) {
      // Only 1 param logic implemented (RX/RY/RZ)
      // Generator Logic corresponding to param index p_idx.
      // For RX/RY/RZ, p_idx=0 is the angle.

      // i. Copy psi to temp
      cmd = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
      copy_buffer(blit, psi, temp, num_qubits);
      [blit endEncoding];

      // ii. Apply Generator P to temp
      // For R(theta), Generator is P.
      // <lambda | P | psi>
      // Note: Factor -i/2 is later.

      mq_gate_t gen_gate;
      gen_gate.num_qubits = 1;
      gen_gate.qubits[0] = gate.qubits[0];
      gen_gate.num_params = 0;

      if (gate.type == MQ_GATE_RX)
        gen_gate.type = MQ_GATE_X;
      else if (gate.type == MQ_GATE_RY)
        gen_gate.type = MQ_GATE_Y;
      else if (gate.type == MQ_GATE_RZ)
        gen_gate.type = MQ_GATE_Z;
      else
        continue; // Unknown param gate type

      enc = [cmd computeCommandEncoder];
      encode_gate(enc, mCtx, num_qubits, &gen_gate, temp);

      // Use GPU Reduction for Dot Product
      [enc setComputePipelineState:reduce_pso];
      [enc setBuffer:lambda offset:0 atIndex:0];
      [enc setBuffer:temp offset:0 atIndex:1];
      [enc setBuffer:partial_sums offset:0 atIndex:2];
      uint32_t total_elems_u32 = (uint32_t)num_elements;
      [enc setBytes:&total_elems_u32 length:sizeof(uint32_t) atIndex:3];

      [enc dispatchThreads:MTLSizeMake(num_elements, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(block_size, 1, 1)];

      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted]; // Must wait for partials

      // Read partial sums from GPU
      mq_complex_t *ps_ptr = (mq_complex_t *)partial_sums.contents;
      double dot_real = 0.0;
      double dot_imag = 0.0;

      // Reduce partials on CPU (Size = grid_size)
      for (uint64_t k = 0; k < grid_size; k++) {
        dot_real += ps_ptr[k].real;
        dot_imag += ps_ptr[k].imag;
      }

      // Output Gradient (Imaginary part of overlap <lambda|P|psi>)
      out_gradients[current_param_index + p_idx] = dot_imag;
    }

    // b. Uncompute psi
    cmd = [queue commandBuffer];
    enc = [cmd computeCommandEncoder];
    encode_gate(enc, mCtx, num_qubits, &inv_gate, psi);

    // c. Backprop lambda
    encode_gate(enc, mCtx, num_qubits, &inv_gate, lambda);

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
  }

  return 0;
}
