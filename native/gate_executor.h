/**
 * gate_executor.h
 */

#import "metalq.h"
#import "state_vector.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@interface MetalQGateExecutor : NSObject

@property(nonatomic, readonly) id<MTLLibrary> library;

- (instancetype)initWithDevice:(id<MTLDevice>)device
                  commandQueue:(id<MTLCommandQueue>)commandQueue;

/**
 * Apply a single gate to the state vector (legacy - creates new command buffer)
 */
- (MetalQError)applyGate:(NSDictionary *)gateData
           toStateVector:(MetalQStateVector *)stateVector;

/**
 * Execute all gates in a single command buffer (batched execution)
 * Much more efficient for circuits with many gates.
 */
- (MetalQError)executeGatesBatched:(NSArray *)gates
                       stateVector:(MetalQStateVector *)stateVector;

@end
