/*
 * gate_executor.m - Batched Gate Execution
 */

#import "context_internal.h"
#import "metalq.h"
#import <Metal/Metal.h>

// Internal Helpers
//
// Thread safety: Python spawns a native call per worker thread, and multiple
// threads may race on the same context's pipeline cache. We synchronize the
// whole function body on the pipelines dictionary (simple, and PSO creation
// is a one-time cost per kernel name over the context's lifetime, so holding
// the lock across it is cheap in steady state).
id<MTLComputePipelineState> get_pipeline(MetalQContext *ctx,
                                         NSString *kernelName) {
  NSMutableDictionary *pipelines =
      (__bridge NSMutableDictionary *)ctx->pipelines;
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)ctx->library;
  id<MTLDevice> device = (__bridge id<MTLDevice>)ctx->device;

  if (!pipelines || !library || !device)
    return nil;

  @synchronized(pipelines) {
    id<MTLComputePipelineState> pso = pipelines[kernelName];
    if (pso)
      return pso;

    id<MTLFunction> kernel = [library newFunctionWithName:kernelName];
    if (!kernel) {
      NSLog(@"[MetalQ] Error: Kernel '%@' not found", kernelName);
      return nil;
    }

    NSError *error = nil;
    pso = [device newComputePipelineStateWithFunction:kernel error:&error];
    if (!pso) {
      NSLog(@"[MetalQ] Error creating PSO for '%@': %@", kernelName, error);
      return nil;
    }

    pipelines[kernelName] = pso;
    return pso;
  }
}

// Fill an interleaved complex 2x2 matrix m[(r*2+c)*2 + re/im]. All matrix
// elements are computed in double (cos/sin, not cosf/sinf, per the precision
// rule) and narrowed to float only at assignment. Returns true on a supported
// gate type, false otherwise (caller fails closed).
bool populate_matrix_single(float *m, mq_gate_type_t type, double *params) {
  for (int i = 0; i < 8; i++)
    m[i] = 0.0f;
  if (type == MQ_GATE_X) {
    m[2] = 1.0f;
    m[4] = 1.0f;
  } else if (type == MQ_GATE_Y) {
    // Y = [[0, -i], [i, 0]]
    // val[1] (0,1) = -i -> real=0, imag=-1
    m[3] = -1.0f;
    // val[2] (1,0) = i  -> real=0, imag=1
    m[5] = 1.0f;
  } else if (type == MQ_GATE_H) {
    double s = 1.0 / sqrt(2.0);
    m[0] = (float)s;
    m[2] = (float)s;
    m[4] = (float)s;
    m[6] = (float)-s;
  } else if (type == MQ_GATE_Z) {
    m[0] = 1.0f;
    m[6] = -1.0f;
  } else if (type == MQ_GATE_RX) {
    double theta = params[0];
    double c = cos(theta / 2);
    double s = sin(theta / 2);
    // [[c, -is], [-is, c]]
    m[0] = (float)c;
    m[3] = (float)-s;
    m[5] = (float)-s;
    m[6] = (float)c;
  } else if (type == MQ_GATE_RY) {
    double theta = params[0];
    double c = cos(theta / 2);
    double s = sin(theta / 2);
    // [[c, -s], [s, c]]
    m[0] = (float)c;
    m[2] = (float)-s;
    m[4] = (float)s;
    m[6] = (float)c;
  } else if (type == MQ_GATE_RZ) {
    double theta = params[0];
    // [[exp(-it/2), 0], [0, exp(it/2)]]
    double c = cos(theta / 2);
    double s = sin(theta / 2);
    m[0] = (float)c;
    m[1] = (float)-s;
    m[6] = (float)c;
    m[7] = (float)s;
  } else if (type == MQ_GATE_S) {
    // diag(1, i)
    m[0] = 1.0f;
    m[7] = 1.0f;
  } else if (type == MQ_GATE_T) {
    // diag(1, exp(i pi/4))
    m[0] = 1.0f;
    m[6] = (float)cos(M_PI / 4);
    m[7] = (float)sin(M_PI / 4);
  } else if (type == MQ_GATE_P || type == MQ_GATE_U1) {
    // diag(1, exp(i theta))
    double theta = params[0];
    m[0] = 1.0f;
    m[6] = (float)cos(theta);
    m[7] = (float)sin(theta);
  } else if (type == MQ_GATE_U3) {
    // U(theta, phi, lam) =
    //   [[cos(t/2),              -e^{i lam} sin(t/2)],
    //    [e^{i phi} sin(t/2),     e^{i(phi+lam)} cos(t/2)]]
    double theta = params[0], phi = params[1], lam = params[2];
    double c = cos(theta / 2);
    double s = sin(theta / 2);
    m[0] = (float)c; // (0,0)
    m[1] = 0.0f;
    m[2] = (float)(-cos(lam) * s); // (0,1) = -e^{i lam} s
    m[3] = (float)(-sin(lam) * s);
    m[4] = (float)(cos(phi) * s); // (1,0) = e^{i phi} s
    m[5] = (float)(sin(phi) * s);
    m[6] = (float)(cos(phi + lam) * c); // (1,1) = e^{i(phi+lam)} c
    m[7] = (float)(sin(phi + lam) * c);
  } else {
    return false;
  }
  return true;
}

bool populate_matrix_two(float *m, mq_gate_type_t type, double *params) {
  for (int i = 0; i < 32; i++)
    m[i] = 0.0f;
// Basis ordering |control target>: row = control*2 + target
#define M_IDX(r, c) ((r * 4 + c) * 2)
  if (type == MQ_GATE_CX) {
    m[M_IDX(0, 0)] = 1.0f;
    m[M_IDX(1, 1)] = 1.0f;
    m[M_IDX(2, 3)] = 1.0f;
    m[M_IDX(3, 2)] = 1.0f;
  } else if (type == MQ_GATE_CY) {
    m[M_IDX(0, 0)] = 1.0f;
    m[M_IDX(1, 1)] = 1.0f;
    m[M_IDX(2, 3) + 1] = -1.0f; // -i
    m[M_IDX(3, 2) + 1] = 1.0f;  // +i
  } else if (type == MQ_GATE_CZ) {
    m[M_IDX(0, 0)] = 1.0f;
    m[M_IDX(1, 1)] = 1.0f;
    m[M_IDX(2, 2)] = 1.0f;
    m[M_IDX(3, 3)] = -1.0f;
  } else if (type == MQ_GATE_SWAP) {
    m[M_IDX(0, 0)] = 1.0f;
    m[M_IDX(1, 2)] = 1.0f;
    m[M_IDX(2, 1)] = 1.0f;
    m[M_IDX(3, 3)] = 1.0f;
  } else if (type == MQ_GATE_CP) {
    double theta = params[0];
    m[M_IDX(0, 0)] = 1.0f;
    m[M_IDX(1, 1)] = 1.0f;
    m[M_IDX(2, 2)] = 1.0f;
    m[M_IDX(3, 3)] = (float)cos(theta);
    m[M_IDX(3, 3) + 1] = (float)sin(theta);
  } else if (type == MQ_GATE_CH) {
    // Controlled-Hadamard: |0c> passes through, |1c> applies H to target.
    // Basis |control target>, row = control*2 + target.
    double s = 1.0 / sqrt(2.0);
    m[M_IDX(0, 0)] = 1.0f;
    m[M_IDX(1, 1)] = 1.0f;
    m[M_IDX(2, 2)] = (float)s;
    m[M_IDX(2, 3)] = (float)s;
    m[M_IDX(3, 2)] = (float)s;
    m[M_IDX(3, 3)] = (float)-s;
  } else {
    return false;
  }
#undef M_IDX
  return true;
}

bool populate_matrix_three(float *m, mq_gate_type_t type, double *params) {
  (void)params;
  for (int i = 0; i < 128; i++)
    m[i] = 0.0f;
// Basis ordering |q0 q1 q2>: row = q0*4 + q1*2 + q2 (matches apply_gate_three)
#define M_IDX(r, c) ((r * 8 + c) * 2)
  if (type == MQ_GATE_CCX) {
    // qubits = (control1, control2, target): swap |110> and |111>
    for (int k = 0; k < 6; k++)
      m[M_IDX(k, k)] = 1.0f;
    m[M_IDX(6, 7)] = 1.0f;
    m[M_IDX(7, 6)] = 1.0f;
  } else if (type == MQ_GATE_CSWAP) {
    // qubits = (control, target1, target2): swap |101> and |110>
    for (int k = 0; k < 8; k++) {
      if (k == 5 || k == 6)
        continue;
      m[M_IDX(k, k)] = 1.0f;
    }
    m[M_IDX(5, 6)] = 1.0f;
    m[M_IDX(6, 5)] = 1.0f;
  } else if (type == MQ_GATE_CCZ) {
    for (int k = 0; k < 8; k++)
      m[M_IDX(k, k)] = (k == 7) ? -1.0f : 1.0f;
  } else {
    return false;
  }
#undef M_IDX
  return true;
}

// out = a * b for interleaved complex 2x2 matrices (m[(r*2+c)*2 + re/im]).
// Gate order: applying gate B then gate A equals the single gate A*B.
static void cmul2x2(const float *a, const float *b, float *out) {
  float tmp[8];
  for (int r = 0; r < 2; r++) {
    for (int c = 0; c < 2; c++) {
      double re = 0.0, im = 0.0;
      for (int k = 0; k < 2; k++) {
        double ar = a[(r * 2 + k) * 2], ai = a[(r * 2 + k) * 2 + 1];
        double br = b[(k * 2 + c) * 2], bi = b[(k * 2 + c) * 2 + 1];
        re += ar * br - ai * bi;
        im += ar * bi + ai * br;
      }
      tmp[(r * 2 + c) * 2] = (float)re;
      tmp[(r * 2 + c) * 2 + 1] = (float)im;
    }
  }
  for (int i = 0; i < 8; i++)
    out[i] = tmp[i];
}

// Dispatch a single-qubit gate given a precomputed 2x2 matrix.
// Returns 0 on success, nonzero on failure (pipeline missing).
static int encode_1q_matrix(id<MTLComputeCommandEncoder> encoder,
                            MetalQContext *ctx, uint32_t num_qubits,
                            uint32_t target, const float *matrix,
                            id<MTLBuffer> stateBuffer) {
  id<MTLComputePipelineState> pso = get_pipeline(ctx, @"apply_gate_single");
  if (!pso)
    return -1;

  [encoder setComputePipelineState:pso];
  [encoder setBuffer:stateBuffer offset:0 atIndex:0];
  [encoder setBytes:&target length:sizeof(uint32_t) atIndex:1];
  [encoder setBytes:matrix length:8 * sizeof(float) atIndex:2];

  uint64_t threads = (1ULL << num_qubits) / 2;
  uint64_t tpg = (threads < 256) ? threads : 256;
  [encoder dispatchThreads:MTLSizeMake(threads, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
  return 0;
}

// Returns 0 on success, nonzero if the gate cannot be encoded (unsupported
// gate type, unsupported qubit count, or missing pipeline). Callers must fail
// closed rather than silently applying identity.
int encode_gate(id<MTLComputeCommandEncoder> encoder, MetalQContext *ctx,
                uint32_t num_qubits, const mq_gate_t *gate,
                id<MTLBuffer> stateBuffer) {

  if (gate->num_qubits == 1) {
    float matrix[8];
    if (!populate_matrix_single(matrix, gate->type, (double *)gate->params)) {
      NSLog(@"[MetalQ] Error: unsupported 1q gate type %d", (int)gate->type);
      return -1;
    }
    return encode_1q_matrix(encoder, ctx, num_qubits, gate->qubits[0], matrix,
                            stateBuffer);
  }

  NSString *kernelName = nil;
  if (gate->num_qubits == 2) {
    kernelName = @"apply_gate_two";
  } else if (gate->num_qubits == 3) {
    kernelName = @"apply_gate_three";
  } else {
    NSLog(@"[MetalQ] Error: %u-qubit gate (type %d) not supported",
          gate->num_qubits, (int)gate->type);
    return -1;
  }

  float matrix2[32];
  float matrix3[128];
  if (gate->num_qubits == 2) {
    if (!populate_matrix_two(matrix2, gate->type, (double *)gate->params)) {
      NSLog(@"[MetalQ] Error: unsupported 2q gate type %d", (int)gate->type);
      return -1;
    }
  } else {
    if (!populate_matrix_three(matrix3, gate->type, (double *)gate->params)) {
      NSLog(@"[MetalQ] Error: unsupported 3q gate type %d", (int)gate->type);
      return -1;
    }
  }

  id<MTLComputePipelineState> pso = get_pipeline(ctx, kernelName);

  if (!pso)
    return -1;

  [encoder setComputePipelineState:pso];
  [encoder setBuffer:stateBuffer offset:0 atIndex:0];

  if (gate->num_qubits == 2) {
    uint32_t control = gate->qubits[0];
    uint32_t target = gate->qubits[1];
    [encoder setBytes:&control length:sizeof(uint32_t) atIndex:1];
    [encoder setBytes:&target length:sizeof(uint32_t) atIndex:2];

    [encoder setBytes:matrix2 length:sizeof(matrix2) atIndex:3];

    uint64_t threads = (1ULL << num_qubits) / 4;
    uint64_t tpg = (threads < 256) ? threads : 256;
    MTLSize gridSize = MTLSizeMake(threads, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(tpg, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];

  } else if (gate->num_qubits == 3) {
    uint32_t qa = gate->qubits[0];
    uint32_t qb = gate->qubits[1];
    uint32_t qc = gate->qubits[2];
    [encoder setBytes:&qa length:sizeof(uint32_t) atIndex:1];
    [encoder setBytes:&qb length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&qc length:sizeof(uint32_t) atIndex:3];

    [encoder setBytes:matrix3 length:sizeof(matrix3) atIndex:4];

    uint64_t threads = (1ULL << num_qubits) / 8;
    uint64_t tpg = (threads < 256) ? threads : 256;
    MTLSize gridSize = MTLSizeMake(threads, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(tpg, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];
  }
  return 0;
}

/**
 * Encode a full gate list, fusing runs of consecutive single-qubit gates on
 * the same qubit into one dispatch (their 2x2 matrices are multiplied on the
 * CPU). Statevector simulation is memory-bandwidth bound, so every avoided
 * dispatch saves a full read+write of the 2^n statevector.
 */
// Returns 0 on success, nonzero if any gate cannot be encoded (fail closed).
int encode_circuit(id<MTLComputeCommandEncoder> encoder, MetalQContext *ctx,
                   uint32_t num_qubits, const mq_gate_t *gates,
                   uint32_t num_gates, id<MTLBuffer> stateBuffer) {
  uint32_t i = 0;
  while (i < num_gates) {
    const mq_gate_t *g = &gates[i];

    if (g->num_qubits == 1) {
      uint32_t target = g->qubits[0];
      float acc[8];
      if (!populate_matrix_single(acc, g->type, (double *)g->params)) {
        NSLog(@"[MetalQ] Error: unsupported 1q gate type %d", (int)g->type);
        return -1;
      }

      uint32_t j = i + 1;
      while (j < num_gates && gates[j].num_qubits == 1 &&
             gates[j].qubits[0] == target) {
        float next[8];
        if (!populate_matrix_single(next, gates[j].type,
                                    (double *)gates[j].params)) {
          NSLog(@"[MetalQ] Error: unsupported 1q gate type %d",
                (int)gates[j].type);
          return -1;
        }
        cmul2x2(next, acc, acc); // acc applied first, then next
        j++;
      }

      if (encode_1q_matrix(encoder, ctx, num_qubits, target, acc,
                           stateBuffer) != 0)
        return -1;
      i = j;
    } else {
      if (encode_gate(encoder, ctx, num_qubits, g, stateBuffer) != 0)
        return -1;
      i++;
    }
  }
  return 0;
}

// Implementation of metalq_run
int metalq_run(mq_context_t ctx, uint32_t num_qubits, const mq_gate_t *gates,
               uint32_t num_gates, uint32_t shots,
               mq_complex_t *out_statevector, uint64_t *out_counts) {
  @autoreleasepool {
    if (!ctx)
      return -1;
    MetalQContext *mCtx = (MetalQContext *)ctx;

    if (!mCtx->device)
      return -1;

    id<MTLDevice> device = (__bridge id<MTLDevice>)mCtx->device;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mCtx->commandQueue;

    if (!device || !queue)
      return -1;

    // 1. Acquire pooled state vector buffer, reset to |0...0> on GPU
    uint64_t sv_bytes = (1ULL << num_qubits) * sizeof(mq_complex_t);
    id<MTLBuffer> svBuffer = mq_pool_acquire(mCtx, device, sv_bytes);
    if (!svBuffer)
      return -2;

    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    encode_statevector_reset(buffer, svBuffer, num_qubits);

    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];

    // 2. Encode Gates (with 1q-gate fusion). Fail closed on any unsupported gate
    // rather than silently applying identity.
    int enc_rc = encode_circuit(encoder, mCtx, num_qubits, gates, num_gates,
                                svBuffer);
    if (enc_rc != 0) {
      [encoder endEncoding];
      mq_pool_release(mCtx, svBuffer);
      return -5;
    }

    [encoder endEncoding];
    [buffer commit];
    [buffer waitUntilCompleted];

    if (buffer.error) {
      printf("[MetalQ] Command Buffer Error: %s\n",
             [[buffer.error description] UTF8String]);
      return -4;
    }

    // 3. Read back results
    if (out_statevector) {
      if (!svBuffer.contents)
        return -3;
      memcpy(out_statevector, svBuffer.contents, sv_bytes);
    }

    mq_pool_release(mCtx, svBuffer);
    return 0;
  }
}
