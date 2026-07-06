/*
 * state_vector.m - GPU State Vector Management
 *
 * Buffer pool + GPU-side statevector initialization. Statevector simulation
 * in a VQE loop allocates the same 2^n buffers thousands of times; pooling
 * them on the context avoids the allocation and first-touch page-fault cost,
 * and blit fills replace full CPU memsets.
 */

#import "context_internal.h"
#import "metalq.h"
#import <Metal/Metal.h>

// Keep at most this many idle buffers per distinct size.
#define MQ_POOL_MAX_PER_SIZE 4

id<MTLBuffer> mq_pool_acquire(MetalQContext *ctx, id<MTLDevice> device,
                              uint64_t length) {
  NSMutableDictionary *pool = (__bridge NSMutableDictionary *)ctx->bufferPool;
  if (pool) {
    @synchronized(pool) {
      NSMutableArray *list = pool[@(length)];
      if (list.count > 0) {
        id<MTLBuffer> buf = list.lastObject;
        [list removeLastObject];
        return buf;
      }
    }
  }
  return [device newBufferWithLength:length
                             options:MTLResourceStorageModeShared];
}

void mq_pool_release(MetalQContext *ctx, id<MTLBuffer> buffer) {
  if (!buffer)
    return;
  NSMutableDictionary *pool = (__bridge NSMutableDictionary *)ctx->bufferPool;
  if (!pool)
    return; // no pool: ARC frees the buffer
  @synchronized(pool) {
    NSMutableArray *list = pool[@(buffer.length)];
    if (!list) {
      list = [NSMutableArray array];
      pool[@(buffer.length)] = list;
    }
    if (list.count < MQ_POOL_MAX_PER_SIZE)
      [list addObject:buffer];
  }
}

void encode_statevector_reset(id<MTLCommandBuffer> cmd, id<MTLBuffer> buf,
                              uint32_t num_qubits) {
  // Buffer is GPU-idle here (pool invariant), so the CPU write is safe and
  // disjoint from the blit-filled range.
  mq_complex_t *ptr = (mq_complex_t *)buf.contents;
  ptr[0].real = 1.0f;
  ptr[0].imag = 0.0f;

  uint64_t len = (1ULL << num_qubits) * sizeof(mq_complex_t);
  if (len > sizeof(mq_complex_t)) {
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit fillBuffer:buf
               range:NSMakeRange(sizeof(mq_complex_t),
                                 len - sizeof(mq_complex_t))
               value:0];
    [blit endEncoding];
  }
}

void encode_zero_fill(id<MTLCommandBuffer> cmd, id<MTLBuffer> buf,
                      uint64_t length) {
  id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
  [blit fillBuffer:buf range:NSMakeRange(0, length) value:0];
  [blit endEncoding];
}
