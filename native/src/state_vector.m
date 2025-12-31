/*
 * state_vector.m - GPU State Vector Management
 */

#import "metalq.h"
#import <Metal/Metal.h>

// Internal helper to manage GPU buffers
// In a full implementation, this would handle synchronisation and memory
// transfer.

id<MTLBuffer> create_statevector_buffer(id<MTLDevice> device,
                                        uint32_t num_qubits) {
  uint64_t num_amplitudes = 1ULL << num_qubits;
  uint64_t size =
      num_amplitudes * sizeof(mq_complex_t); // 16 bytes per amplitude

  // Use Shared storage mode for easy CPU access on Apple Silicon
  MTLResourceOptions options = MTLResourceStorageModeShared;

  id<MTLBuffer> buffer = [device newBufferWithLength:size options:options];
  if (!buffer)
    return nil;

  // Initialize to |0...0>
  mq_complex_t *ptr = (mq_complex_t *)buffer.contents;
  memset(ptr, 0, size);
  ptr[0].real = 1.0f;
  ptr[0].imag = 0.0f;

  return buffer;
}
