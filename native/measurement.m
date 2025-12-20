/**
 * measurement.m - GPU-accelerated Measurement Implementation
 */

#import "measurement.h"
#import <stdlib.h>

@implementation MetalQMeasurement {
  id<MTLDevice> _device;
  id<MTLCommandQueue> _commandQueue;
  id<MTLLibrary> _library;

  // GPU Pipeline States
  id<MTLComputePipelineState> _probPipeline;
  id<MTLComputePipelineState> _prefixUpPipeline;
  id<MTLComputePipelineState> _prefixDownPipeline;
  id<MTLComputePipelineState> _sampleCountPipeline;
  id<MTLComputePipelineState> _normalizePipeline;
  id<MTLComputePipelineState> _zeroHistPipeline;

  BOOL _gpuEnabled;
}

#pragma mark - Initialization

- (instancetype)initWithDevice:(id<MTLDevice>)device
                  commandQueue:(id<MTLCommandQueue>)commandQueue
                       library:(id<MTLLibrary>)library {
  self = [super init];
  if (self) {
    _device = device;
    _commandQueue = commandQueue;
    _library = library;
    _gpuEnabled = NO;

    if (library) {
      _gpuEnabled = [self _createPipelines];
    }

    if (_gpuEnabled) {
      NSLog(@"Metal-Q: GPU sampling enabled");
    } else {
      NSLog(@"Metal-Q: GPU sampling unavailable, using CPU fallback");
    }
  }
  return self;
}

- (BOOL)_createPipelines {
  NSError *error = nil;

  id<MTLFunction> probFunc =
      [_library newFunctionWithName:@"compute_probabilities"];
  id<MTLFunction> upFunc =
      [_library newFunctionWithName:@"prefix_sum_up_sweep"];
  id<MTLFunction> downFunc =
      [_library newFunctionWithName:@"prefix_sum_down_sweep"];
  id<MTLFunction> sampleFunc =
      [_library newFunctionWithName:@"sample_and_count"];
  id<MTLFunction> normFunc =
      [_library newFunctionWithName:@"normalize_cum_prob"];
  id<MTLFunction> zeroFunc = [_library newFunctionWithName:@"zero_histogram"];

  if (!probFunc || !upFunc || !downFunc || !sampleFunc) {
    NSLog(@"Metal-Q: Missing measurement shader functions");
    return NO;
  }

  _probPipeline = [_device newComputePipelineStateWithFunction:probFunc
                                                         error:&error];
  _prefixUpPipeline = [_device newComputePipelineStateWithFunction:upFunc
                                                             error:&error];
  _prefixDownPipeline = [_device newComputePipelineStateWithFunction:downFunc
                                                               error:&error];
  _sampleCountPipeline = [_device newComputePipelineStateWithFunction:sampleFunc
                                                                error:&error];

  if (normFunc) {
    _normalizePipeline = [_device newComputePipelineStateWithFunction:normFunc
                                                                error:&error];
  }
  if (zeroFunc) {
    _zeroHistPipeline = [_device newComputePipelineStateWithFunction:zeroFunc
                                                               error:&error];
  }

  if (!_probPipeline || !_prefixUpPipeline || !_prefixDownPipeline ||
      !_sampleCountPipeline) {
    NSLog(@"Metal-Q: Failed to create measurement pipelines: %@", error);
    return NO;
  }

  return YES;
}

#pragma mark - GPU Sampling

- (MetalQError)sampleFromStateVector:(MetalQStateVector *)stateVector
                        measurements:(NSArray *)measurements
                           numClbits:(int)numClbits
                               shots:(int)shots
                             results:(NSMutableDictionary *)results {

  // For small problems or if GPU not available, use CPU
  if (!_gpuEnabled || stateVector.size < 4096 || shots < 256) {
    return [MetalQMeasurement sampleFromStateVectorCPU:stateVector
                                          measurements:measurements
                                             numClbits:numClbits
                                                 shots:shots
                                               results:results];
  }

  NSUInteger size = stateVector.size;

  // Step 1: Compute cumulative probabilities on CPU (O(n) - fast enough)
  float *real = (float *)[stateVector.realBuffer contents];
  float *imag = (float *)[stateVector.imagBuffer contents];

  id<MTLBuffer> cumProbBuffer =
      [_device newBufferWithLength:size * sizeof(float)
                           options:MTLResourceStorageModeShared];
  id<MTLBuffer> histogramBuffer =
      [_device newBufferWithLength:size * sizeof(uint32_t)
                           options:MTLResourceStorageModeShared];

  if (!cumProbBuffer || !histogramBuffer) {
    return METALQ_ERROR_OUT_OF_MEMORY;
  }

  // Compute cumulative probabilities on CPU (simple and fast)
  float *cumProb = (float *)[cumProbBuffer contents];
  float totalProb = 0.0f;

  for (NSUInteger i = 0; i < size; i++) {
    float prob = real[i] * real[i] + imag[i] * imag[i];
    totalProb += prob;
    cumProb[i] = totalProb;
  }

  // Normalize
  if (totalProb > 0) {
    for (NSUInteger i = 0; i < size; i++) {
      cumProb[i] /= totalProb;
    }
  }

  // Zero histogram buffer
  memset([histogramBuffer contents], 0, size * sizeof(uint32_t));

  // Step 2: Sample and count on GPU (parallel binary search + atomic histogram)
  [self _sampleAndCount:cumProbBuffer
              histogram:histogramBuffer
                   size:size
                  shots:shots];

  // Step 3: Extract results from histogram
  [self _extractResults:histogramBuffer
                   size:size
           measurements:measurements
              numClbits:numClbits
                results:results];

  return METALQ_SUCCESS;
}

#pragma mark - GPU Kernels

- (void)_computeProbabilities:(MetalQStateVector *)sv
                       output:(id<MTLBuffer>)probBuffer
                         size:(NSUInteger)size {

  id<MTLCommandBuffer> cmdBuffer = [_commandQueue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

  [encoder setComputePipelineState:_probPipeline];
  [encoder setBuffer:sv.realBuffer offset:0 atIndex:0];
  [encoder setBuffer:sv.imagBuffer offset:0 atIndex:1];
  [encoder setBuffer:probBuffer offset:0 atIndex:2];

  uint32_t sizeVal = (uint32_t)size;
  [encoder setBytes:&sizeVal length:sizeof(uint32_t) atIndex:3];

  MTLSize gridSize = MTLSizeMake(size, 1, 1);
  MTLSize threadGroupSize =
      MTLSizeMake(MIN(256, _probPipeline.maxTotalThreadsPerThreadgroup), 1, 1);

  [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
  [encoder endEncoding];

  [cmdBuffer commit];
  [cmdBuffer waitUntilCompleted];
}

- (void)_inclusivePrefixSum:(id<MTLBuffer>)buffer size:(NSUInteger)size {
  // Blelloch inclusive scan algorithm
  // Up-sweep (reduce) phase
  for (NSUInteger stride = 1; stride < size; stride *= 2) {
    NSUInteger numThreads = size / (stride * 2);
    if (numThreads == 0)
      break;

    id<MTLCommandBuffer> cmdBuffer = [_commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

    [encoder setComputePipelineState:_prefixUpPipeline];
    [encoder setBuffer:buffer offset:0 atIndex:0];

    uint32_t strideVal = (uint32_t)stride;
    uint32_t sizeVal = (uint32_t)size;
    [encoder setBytes:&strideVal length:sizeof(uint32_t) atIndex:1];
    [encoder setBytes:&sizeVal length:sizeof(uint32_t) atIndex:2];

    MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(MIN(256, numThreads), 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];

    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
  }

  // Down-sweep phase (for inclusive scan, slightly different from exclusive)
  for (NSUInteger stride = size / 4; stride >= 1; stride /= 2) {
    NSUInteger numThreads = size / (stride * 2);
    if (numThreads == 0)
      numThreads = 1;

    id<MTLCommandBuffer> cmdBuffer = [_commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

    [encoder setComputePipelineState:_prefixDownPipeline];
    [encoder setBuffer:buffer offset:0 atIndex:0];

    uint32_t strideVal = (uint32_t)stride;
    uint32_t sizeVal = (uint32_t)size;
    [encoder setBytes:&strideVal length:sizeof(uint32_t) atIndex:1];
    [encoder setBytes:&sizeVal length:sizeof(uint32_t) atIndex:2];

    MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(MIN(256, numThreads), 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];

    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
  }
}

- (void)_sampleAndCount:(id<MTLBuffer>)cumProbBuffer
              histogram:(id<MTLBuffer>)histogramBuffer
                   size:(NSUInteger)size
                  shots:(int)shots {

  // Configure thread count - each thread handles multiple shots
  NSUInteger numThreads = MIN(1024, (NSUInteger)shots);
  uint32_t shotsPerThread =
      ((uint32_t)shots + (uint32_t)numThreads - 1) / (uint32_t)numThreads;

  id<MTLCommandBuffer> cmdBuffer = [_commandQueue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

  [encoder setComputePipelineState:_sampleCountPipeline];
  [encoder setBuffer:cumProbBuffer offset:0 atIndex:0];
  [encoder setBuffer:histogramBuffer offset:0 atIndex:1];

  uint32_t sizeVal = (uint32_t)size;
  uint32_t seed = arc4random();

  [encoder setBytes:&sizeVal length:sizeof(uint32_t) atIndex:2];
  [encoder setBytes:&seed length:sizeof(uint32_t) atIndex:3];
  [encoder setBytes:&shotsPerThread length:sizeof(uint32_t) atIndex:4];

  MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
  MTLSize threadGroupSize = MTLSizeMake(MIN(256, numThreads), 1, 1);

  [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
  [encoder endEncoding];

  [cmdBuffer commit];
  [cmdBuffer waitUntilCompleted];
}

- (void)_extractResults:(id<MTLBuffer>)histogramBuffer
                   size:(NSUInteger)size
           measurements:(NSArray *)measurements
              numClbits:(int)numClbits
                results:(NSMutableDictionary *)results {

  uint32_t *histogram = (uint32_t *)[histogramBuffer contents];

  // Extract measurement mappings
  NSMutableArray *qubitIndices = [NSMutableArray array];
  NSMutableArray *clbitIndices = [NSMutableArray array];

  for (NSArray *m in measurements) {
    [qubitIndices addObject:m[0]];
    [clbitIndices addObject:m[1]];
  }

  // Convert histogram to bitstring counts
  for (NSUInteger state = 0; state < size; state++) {
    uint32_t count = histogram[state];
    if (count == 0)
      continue;

    // Construct bitstring from state
    char *bitstring = (char *)calloc(numClbits + 1, sizeof(char));
    memset(bitstring, '0', numClbits);

    for (NSUInteger i = 0; i < [qubitIndices count]; i++) {
      int qubit = [qubitIndices[i] intValue];
      int clbit = [clbitIndices[i] intValue];

      int bit = (state >> qubit) & 1;
      bitstring[numClbits - 1 - clbit] = bit ? '1' : '0';
    }

    NSString *key = [NSString stringWithUTF8String:bitstring];
    free(bitstring);

    results[key] = @(count);
  }
}

#pragma mark - CPU Fallback

+ (MetalQError)sampleFromStateVectorCPU:(MetalQStateVector *)stateVector
                           measurements:(NSArray *)measurements
                              numClbits:(int)numClbits
                                  shots:(int)shots
                                results:(NSMutableDictionary *)results {

  NSUInteger size = stateVector.size;
  float *real = (float *)[stateVector.realBuffer contents];
  float *imag = (float *)[stateVector.imagBuffer contents];

  // Calculate cumulative probability distribution
  double *cumProb = (double *)malloc(size * sizeof(double));
  if (!cumProb)
    return METALQ_ERROR_OUT_OF_MEMORY;

  double totalProb = 0.0;

  for (NSUInteger i = 0; i < size; i++) {
    double prob = (double)(real[i] * real[i] + imag[i] * imag[i]);
    totalProb += prob;
    cumProb[i] = totalProb;
  }

  // Normalize
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

      int bit = (state >> qubit) & 1;
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
