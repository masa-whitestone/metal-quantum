/**
 * state_vector.h
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@interface MetalQStateVector : NSObject

@property(nonatomic, readonly) int numQubits;
@property(nonatomic, readonly) NSUInteger size; // 2^numQubits
@property(nonatomic, readonly) id<MTLBuffer> realBuffer;
@property(nonatomic, readonly) id<MTLBuffer> imagBuffer;

- (instancetype)initWithDevice:(id<MTLDevice>)device numQubits:(int)numQubits;

/**
 * Reset to |0...0> state
 */
- (void)reset;

/**
 * Copy to host memory
 */
- (void)copyToHost:(float *)real imaginary:(float *)imag;

@end
