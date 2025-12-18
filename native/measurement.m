/**
 * measurement.m
 */

#import "measurement.h"
#import <stdlib.h>

@implementation MetalQMeasurement

+ (MetalQError)sampleFromStateVector:(MetalQStateVector *)stateVector
                        measurements:(NSArray *)measurements
                           numClbits:(int)numClbits
                               shots:(int)shots
                             results:(NSMutableDictionary *)results {

  NSUInteger size = stateVector.size;
  float *real = (float *)[stateVector.realBuffer contents];
  float *imag = (float *)[stateVector.imagBuffer contents];

  // Calculate cumulative probability distribution
  // Note: for large states, this should arguably be done on GPU (parallel
  // reductions) but for MVP doing this on CPU is fine up to ~20 qubits.

  double *cumProb = (double *)malloc(size * sizeof(double));
  if (!cumProb)
    return METALQ_ERROR_OUT_OF_MEMORY;

  double totalProb = 0.0;

  for (NSUInteger i = 0; i < size; i++) {
    double prob = (double)(real[i] * real[i] + imag[i] * imag[i]);
    totalProb += prob;
    cumProb[i] = totalProb;
  }

  // Normalize (should be close to 1.0 but floats drift)
  for (NSUInteger i = 0; i < size; i++) {
    cumProb[i] /= totalProb;
  }

  // Extract measurement targets
  NSMutableArray *qubitIndices = [NSMutableArray array];
  NSMutableArray *clbitIndices = [NSMutableArray array];

  for (NSArray *m in measurements) {
    [qubitIndices addObject:m[0]];
    [clbitIndices addObject:m[1]];
  }

  // Sampling loop
  for (int shot = 0; shot < shots; shot++) {
    double r = (double)arc4random() / (double)UINT32_MAX;

    // Binary search
    NSUInteger state = 0;
    NSUInteger low = 0, high = size - 1;
    while (low < high) {
      NSUInteger mid = (low + high) / 2;
      if (cumProb[mid] < r) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    state = low;

    // Construct bitstring
    char *bitstring = (char *)calloc(numClbits + 1, sizeof(char));
    memset(bitstring, '0', numClbits);

    for (NSUInteger i = 0; i < [qubitIndices count]; i++) {
      int qubit = [qubitIndices[i] intValue];
      int clbit = [clbitIndices[i] intValue];

      // Extract bit from state index
      int bit = (state >> qubit) & 1;
      // Qiskit bitstring order is little-endian (clbit 0 is rightmost)
      bitstring[numClbits - 1 - clbit] = bit ? '1' : '0';
    }

    NSString *key = [NSString stringWithUTF8String:bitstring];
    free(bitstring);

    NSNumber *count = results[key];
    results[key] = @(count ? [count intValue] + 1 : 1);
  }

  free(cumProb);
  return METALQ_SUCCESS;
}

@end
