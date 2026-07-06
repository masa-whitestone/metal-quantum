/*
 * context_internal.h - Internal context structure sharing
 */

#ifndef CONTEXT_INTERNAL_H
#define CONTEXT_INTERNAL_H

#import "metalq.h" // Needed for mq_gate_t
#import <Metal/Metal.h>

typedef struct {
  void *device;       // id<MTLDevice>
  void *commandQueue; // id<MTLCommandQueue>
  void *library;      // id<MTLLibrary>
  void *pipelines;    // NSMutableDictionary*
  void *bufferPool;   // NSMutableDictionary* (length -> NSMutableArray of MTLBuffer)
} MetalQContext;

// Buffer pool (state_vector.m). Buffers handed back via mq_pool_release must
// be GPU-idle (callers waitUntilCompleted before releasing).
id<MTLBuffer> mq_pool_acquire(MetalQContext *ctx, id<MTLDevice> device,
                              uint64_t length);
void mq_pool_release(MetalQContext *ctx, id<MTLBuffer> buffer);

// Statevector (re)initialization. CPU writes amplitude 0, GPU blit-fills the
// rest, so pooled buffers never pay a full CPU memset. The buffer must be
// GPU-idle when called (the CPU write happens at encode time).
void encode_statevector_reset(id<MTLCommandBuffer> cmd, id<MTLBuffer> buf,
                              uint32_t num_qubits);
void encode_zero_fill(id<MTLCommandBuffer> cmd, id<MTLBuffer> buf,
                      uint64_t length);

// Per-term parameters for the fused Pauli-string kernels.
// Must match PauliTermParams in adjoint.metal.
typedef struct {
  uint32_t x_mask;
  uint32_t y_mask;
  uint32_t z_mask;
  uint32_t total_elements;
  float g_re;
  float g_im;
} mq_pauli_params_t;

// Decompose Pauli term j of an mq_hamiltonian_t into bit masks and the global
// phase g = (-i)^nY, so <i|P|i^x> = g * (-1)^popcount(i & (y|z)).
static inline void mq_term_masks(const mq_hamiltonian_t *H, uint32_t j,
                                 uint32_t *x_mask, uint32_t *y_mask,
                                 uint32_t *z_mask, float *g_re, float *g_im) {
  uint32_t x = 0, y = 0, z = 0;
  for (uint32_t q = 0; q < H->num_qubits; q++) {
    uint8_t code = H->pauli_codes[j * H->num_qubits + q];
    if (code == 1)
      x |= (1u << q);
    else if (code == 2) {
      x |= (1u << q);
      y |= (1u << q);
    } else if (code == 3)
      z |= (1u << q);
  }
  static const float G_RE[4] = {1.0f, 0.0f, -1.0f, 0.0f};
  static const float G_IM[4] = {0.0f, -1.0f, 0.0f, 1.0f};
  uint32_t ny = (uint32_t)__builtin_popcount(y) & 3u;
  *x_mask = x;
  *y_mask = y;
  *z_mask = z;
  *g_re = G_RE[ny];
  *g_im = G_IM[ny];
}
// Both return 0 on success, nonzero if a gate cannot be encoded (fail closed).
int encode_gate(id<MTLComputeCommandEncoder> encoder, MetalQContext *ctx,
                uint32_t num_qubits, const mq_gate_t *gate,
                id<MTLBuffer> stateBuffer);
int encode_circuit(id<MTLComputeCommandEncoder> encoder, MetalQContext *ctx,
                   uint32_t num_qubits, const mq_gate_t *gates,
                   uint32_t num_gates, id<MTLBuffer> stateBuffer);
id<MTLComputePipelineState> get_pipeline(MetalQContext *ctx,
                                         NSString *kernelName);

#endif
