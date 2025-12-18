/**
 * state_vector.m
 */

#import "state_vector.h"

@implementation MetalQStateVector {
  id<MTLDevice> _device;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device numQubits:(int)numQubits {
  self = [super init];
  if (self) {
    _device = device;
    _numQubits = numQubits;
    _size = 1UL << numQubits; // 2^n

    // Allocate GPU buffers
    NSUInteger bufferSize = _size * sizeof(float);

    _realBuffer = [device newBufferWithLength:bufferSize
                                      options:MTLResourceStorageModeShared];
    _imagBuffer = [device newBufferWithLength:bufferSize
                                      options:MTLResourceStorageModeShared];

    if (!_realBuffer || !_imagBuffer) {
      NSLog(@"Metal-Q: Failed to allocate state vector buffers");
      return nil;
    }

    [self reset];
  }
  return self;
}

- (void)reset {
  // |0...0> = [1, 0, 0, ..., 0]
  float *real = (float *)[_realBuffer contents];
  float *imag = (float *)[_imagBuffer contents];

  memset(real, 0, _size * sizeof(float));
  memset(imag, 0, _size * sizeof(float));

  real[0] = 1.0f;
}

- (void)copyToHost:(float *)real imaginary:(float *)imag {
  memcpy(real, [_realBuffer contents], _size * sizeof(float));
  memcpy(imag, [_imagBuffer contents], _size * sizeof(float));
}

@end
