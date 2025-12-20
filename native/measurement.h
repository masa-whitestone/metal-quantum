/**
 * measurement.h - GPU-accelerated Measurement
 */

#import "metalq.h"
#import "state_vector.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@interface MetalQMeasurement : NSObject

/**
 * Initialize with Metal device for GPU-accelerated sampling
 */
- (instancetype)initWithDevice:(id<MTLDevice>)device
                  commandQueue:(id<MTLCommandQueue>)commandQueue
                       library:(id<MTLLibrary>)library;

/**
 * GPU-accelerated sampling from state vector
 */
- (MetalQError)sampleFromStateVector:(MetalQStateVector *)stateVector
                        measurements:(NSArray *)measurements
                           numClbits:(int)numClbits
                               shots:(int)shots
                             results:(NSMutableDictionary *)results;

/**
 * CPU fallback (class method for when GPU is unavailable)
 */
+ (MetalQError)sampleFromStateVectorCPU:(MetalQStateVector *)stateVector
                           measurements:(NSArray *)measurements
                              numClbits:(int)numClbits
                                  shots:(int)shots
                                results:(NSMutableDictionary *)results;

@end
