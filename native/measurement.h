/**
 * measurement.h
 */

#import "metalq.h"
#import "state_vector.h"
#import <Foundation/Foundation.h>

@interface MetalQMeasurement : NSObject

/**
 * Sample from state vector
 *
 * @param stateVector State vector
 * @param measurements Measurement list [(qubit, clbit), ...]
 * @param numClbits Number of classical bits
 * @param shots Number of shots
 * @param results Results dictionary (output)
 */
+ (MetalQError)sampleFromStateVector:(MetalQStateVector *)stateVector
                        measurements:(NSArray *)measurements
                           numClbits:(int)numClbits
                               shots:(int)shots
                             results:(NSMutableDictionary *)results;

@end
