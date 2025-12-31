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
} MetalQContext;

// Helper function declarations
id<MTLBuffer> create_statevector_buffer(id<MTLDevice> device,
                                        uint32_t num_qubits);
void encode_gate(id<MTLComputeCommandEncoder> encoder, MetalQContext *ctx,
                 uint32_t num_qubits, const mq_gate_t *gate,
                 id<MTLBuffer> stateBuffer);

#endif
