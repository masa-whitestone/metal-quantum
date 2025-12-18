/**
 * gate_executor.h
 */

#import "metalq.h"
#import "state_vector.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@interface MetalQGateExecutor : NSObject

- (instancetype)initWithDevice:(id<MTLDevice>)device
                  commandQueue:(id<MTLCommandQueue>)commandQueue;

/**
 * Apply a gate to the state vector
 *
 * @param gateData Gate data (JSON dictionary)
 * @param stateVector Target state vector
 * @return Error code
 */
- (MetalQError)applyGate:(NSDictionary *)gateData
           toStateVector:(MetalQStateVector *)stateVector;

@end
