/*
 * gate_executor.m - Batched Gate Execution
 */

#import "context_internal.h"
#import "metalq.h"
#import <Metal/Metal.h>

// Internal Helpers
id<MTLComputePipelineState> get_pipeline(MetalQContext *ctx,
                                         NSString *kernelName) {
  NSMutableDictionary *pipelines =
      (__bridge NSMutableDictionary *)ctx->pipelines;
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)ctx->library;
  id<MTLDevice> device = (__bridge id<MTLDevice>)ctx->device;

  if (!pipelines || !library || !device)
    return nil;

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

void populate_matrix_single(float *m, mq_gate_type_t type, double *params) {
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
    float s = 1.0f / sqrtf(2.0f);
    m[0] = s;
    m[2] = s;
    m[4] = s;
    m[6] = -s;
  } else if (type == MQ_GATE_Z) {
    m[0] = 1.0f;
    m[6] = -1.0f;
  } else if (type == MQ_GATE_RX) {
    double theta = params[0];
    float c = cosf(theta / 2);
    float s = sinf(theta / 2);
    // [[c, -is], [-is, c]]
    m[0] = c;
    m[1] = 0;
    m[2] = 0;
    m[3] = -s;
    m[4] = 0;
    m[5] = -s;
    m[6] = c;
    m[7] = 0;
  } else if (type == MQ_GATE_RY) {
    double theta = params[0];
    float c = cosf(theta / 2);
    float s = sinf(theta / 2);
    // [[c, -s], [s, c]]
    m[0] = c;
    m[1] = 0;
    m[2] = -s;
    m[3] = 0;
    m[4] = s;
    m[5] = 0;
    m[6] = c;
    m[7] = 0;
  } else if (type == MQ_GATE_RZ) {
    double theta = params[0];
    // [[exp(-it/2), 0], [0, exp(it/2)]]
    // exp(-it/2) = cos(t/2) - i sin(t/2)
    // exp(it/2)  = cos(t/2) + i sin(t/2)
    m[0] = cosf(theta / 2);
    m[1] = -sinf(theta / 2);
    m[6] = cosf(theta / 2);
    m[7] = sinf(theta / 2);
  } else {
    m[0] = 1.0f;
    m[6] = 1.0f;
  }
}

void populate_matrix_two(float *m, mq_gate_type_t type, double *params) {
  for (int i = 0; i < 32; i++)
    m[i] = 0.0f;
#define M_IDX(r, c) ((r * 4 + c) * 2)
  if (type == MQ_GATE_CX) {
    m[M_IDX(0, 0)] = 1.0f;
    m[M_IDX(1, 1)] = 1.0f;
    m[M_IDX(2, 3)] = 1.0f;
    m[M_IDX(3, 2)] = 1.0f;
  } else {
    m[M_IDX(0, 0)] = 1.0f;
    m[M_IDX(1, 1)] = 1.0f;
    m[M_IDX(2, 2)] = 1.0f;
    m[M_IDX(3, 3)] = 1.0f;
  }
}

void encode_gate(id<MTLComputeCommandEncoder> encoder, MetalQContext *ctx,
                 uint32_t num_qubits, const mq_gate_t *gate,
                 id<MTLBuffer> stateBuffer) {

  NSString *kernelName = nil;
  if (gate->num_qubits == 1) {
    kernelName = @"apply_gate_single";
  } else if (gate->num_qubits == 2) {
    kernelName = @"apply_gate_two";
  } else {
    return;
  }

  id<MTLComputePipelineState> pso = get_pipeline(ctx, kernelName);

  if (!pso)
    return;

  [encoder setComputePipelineState:pso];
  [encoder setBuffer:stateBuffer offset:0 atIndex:0];

  if (gate->num_qubits == 1) {
    uint32_t target = gate->qubits[0];
    [encoder setBytes:&target length:sizeof(uint32_t) atIndex:1];

    float matrix[8];
    populate_matrix_single(matrix, gate->type, (double *)gate->params);
    [encoder setBytes:matrix length:sizeof(matrix) atIndex:2];

    uint64_t threads = (1ULL << num_qubits) / 2;
    uint64_t tpg = (threads < 256) ? threads : 256;
    MTLSize gridSize = MTLSizeMake(threads, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(tpg, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];

  } else if (gate->num_qubits == 2) {
    uint32_t control = gate->qubits[0];
    uint32_t target = gate->qubits[1];
    [encoder setBytes:&control length:sizeof(uint32_t) atIndex:1];
    [encoder setBytes:&target length:sizeof(uint32_t) atIndex:2];

    float matrix[32];
    populate_matrix_two(matrix, gate->type, (double *)gate->params);
    [encoder setBytes:matrix length:sizeof(matrix) atIndex:3];

    uint64_t threads = (1ULL << num_qubits) / 4;
    uint64_t tpg = (threads < 256) ? threads : 256;
    MTLSize gridSize = MTLSizeMake(threads, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(tpg, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];
  }
}

// Implementation of metalq_run
int metalq_run(mq_context_t ctx, uint32_t num_qubits, const mq_gate_t *gates,
               uint32_t num_gates, uint32_t shots,
               mq_complex_t *out_statevector, uint64_t *out_counts) {

  if (!ctx)
    return -1;
  MetalQContext *mCtx = (MetalQContext *)ctx;

  if (!mCtx->device)
    return -1;

  id<MTLDevice> device = (__bridge id<MTLDevice>)mCtx->device;
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mCtx->commandQueue;

  if (!device || !queue)
    return -1;

  // 1. Create State Vector Buffer
  id<MTLBuffer> svBuffer = create_statevector_buffer(device, num_qubits);
  if (!svBuffer)
    return -2;

  id<MTLCommandBuffer> buffer = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];

  // 2. Encode Gates
  for (uint32_t i = 0; i < num_gates; i++) {
    encode_gate(encoder, mCtx, num_qubits, &gates[i], svBuffer);
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
    uint64_t len = (1ULL << num_qubits) * sizeof(mq_complex_t);
    if (!svBuffer.contents)
      return -3;
    memcpy(out_statevector, svBuffer.contents, len);
  }

  return 0;
}
