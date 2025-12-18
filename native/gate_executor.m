/**
 * gate_executor.m
 */

#import "gate_executor.h"
#include <dlfcn.h>
#import <simd/simd.h>

// Shared structures with Metal Shaders
typedef struct {
  simd_float2 matrix[4]; // 2x2 complex matrix (each element is [real, imag])
} GateMatrix1Q;

typedef struct {
  simd_float2 matrix[16]; // 4x4 complex matrix
} GateMatrix2Q;

typedef struct {
  uint32_t targetQubit;
  uint32_t controlQubit; // for controlled gates
  uint32_t numQubits;
  uint32_t stateSize;
} GateParams;

@interface MetalQGateExecutor (Private)
- (BOOL)_createPipelines;
@end

@implementation MetalQGateExecutor {
  id<MTLDevice> _device;
  id<MTLCommandQueue> _commandQueue;
  id<MTLLibrary> _library;

  // Pipeline states
  id<MTLComputePipelineState> _gate1QPipeline;
  id<MTLComputePipelineState> _gate2QPipeline;
  id<MTLComputePipelineState> _controlledGatePipeline;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device
                  commandQueue:(id<MTLCommandQueue>)commandQueue {
  self = [super init];
  if (self) {
    _device = device;
    _commandQueue = commandQueue;

    if (![self _loadShaders]) {
      return nil;
    }
  }
  return self;
}

- (BOOL)_loadShaders {
  NSError *error = nil;

  // Strategy 1: Find metallib relative to this dylib
  Dl_info info;
  // Assuming metalq_init is a function symbol defined elsewhere in the dylib
  extern int metalq_init(void);
  if (dladdr((void *)metalq_init, &info)) {
    NSString *dylibPath = [NSString stringWithUTF8String:info.dli_fname];
    NSString *dirPath = [dylibPath stringByDeletingLastPathComponent];
    NSString *metallibPath =
        [dirPath stringByAppendingPathComponent:@"quantum_gates.metallib"];

    if ([[NSFileManager defaultManager] fileExistsAtPath:metallibPath]) {
      NSURL *url = [NSURL fileURLWithPath:metallibPath];
      _library = [_device newLibraryWithURL:url error:&error];
      if (_library) {
        // NSLog(@"Loaded library from dylib dir: %@", metallibPath);
        return [self _createPipelines];
      }
    }
  }

  // Strategy 2: Fallback to search paths (legacy/dev)
  NSArray *searchPaths = @[
    @"quantum_gates.metallib", @"lib/quantum_gates.metallib",
    @"../lib/quantum_gates.metallib", @"build/quantum_gates.metallib"
  ];

  for (NSString *path in searchPaths) {
    if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
      NSURL *url = [NSURL fileURLWithPath:path];
      _library = [_device newLibraryWithURL:url error:&error];
      if (_library) {
        // NSLog(@"Loaded library from %@", path);
        return [self _createPipelines];
      }
    }
  }

  NSLog(@"Metal-Q: Failed to load shader library.");
  return NO;
}

- (BOOL)_createPipelines {
  NSError *error = nil;
  id<MTLFunction> gate1QFunc = [_library newFunctionWithName:@"apply_gate_1q"];
  id<MTLFunction> gate2QFunc = [_library newFunctionWithName:@"apply_gate_2q"];
  id<MTLFunction> ctrlGateFunc =
      [_library newFunctionWithName:@"apply_controlled_gate"];

  if (!gate1QFunc || !gate2QFunc || !ctrlGateFunc) {
    NSLog(@"Metal-Q: Failed to find shader functions");
    return NO;
  }

  _gate1QPipeline = [_device newComputePipelineStateWithFunction:gate1QFunc
                                                           error:&error];
  _gate2QPipeline = [_device newComputePipelineStateWithFunction:gate2QFunc
                                                           error:&error];
  _controlledGatePipeline =
      [_device newComputePipelineStateWithFunction:ctrlGateFunc error:&error];

  if (!_gate1QPipeline || !_gate2QPipeline || !_controlledGatePipeline) {
    NSLog(@"Metal-Q: Failed to create compute pipelines: %@", error);
    return NO;
  }
  return YES;
}

- (MetalQError)applyGate:(NSDictionary *)gateData
           toStateVector:(MetalQStateVector *)stateVector {

  NSString *gateName = gateData[@"name"];
  NSArray *qubits = gateData[@"qubits"];
  NSArray *params = gateData[@"params"];

  GateMatrix1Q matrix1Q;
  GateMatrix2Q matrix2Q;

  if ([qubits count] == 1) {
    [self _getGateMatrix1Q:gateName params:params result:&matrix1Q];
    return [self _apply1QGate:&matrix1Q
                       target:[qubits[0] intValue]
                  stateVector:stateVector];
  } else if ([qubits count] == 2) {
    if ([self _isControlledGate:gateName]) {
      [self _getGateMatrix1Q:[self _baseGateName:gateName]
                      params:params
                      result:&matrix1Q];
      return [self _applyControlledGate:&matrix1Q
                                control:[qubits[0] intValue]
                                 target:[qubits[1] intValue]
                            stateVector:stateVector];
    } else {
      [self _getGateMatrix2Q:gateName params:params result:&matrix2Q];
      return [self _apply2QGate:&matrix2Q
                         qubit0:[qubits[0] intValue]
                         qubit1:[qubits[1] intValue]
                    stateVector:stateVector];
    }
  }

  return METALQ_ERROR_UNSUPPORTED_GATE;
}

#pragma mark - Gate Matrix Helpers

- (void)_getGateMatrix1Q:(NSString *)name
                  params:(NSArray *)params
                  result:(GateMatrix1Q *)matrix {

  float r2 = 1.0f / sqrtf(2.0f);

  // Initialize defaults to I (Identity)
  matrix->matrix[0] = simd_make_float2(1, 0);
  matrix->matrix[1] = simd_make_float2(0, 0);
  matrix->matrix[2] = simd_make_float2(0, 0);
  matrix->matrix[3] = simd_make_float2(1, 0);

  if ([name isEqualToString:@"id"])
    return;

  if ([name isEqualToString:@"x"]) {
    matrix->matrix[0] = simd_make_float2(0, 0);
    matrix->matrix[1] = simd_make_float2(1, 0);
    matrix->matrix[2] = simd_make_float2(1, 0);
    matrix->matrix[3] = simd_make_float2(0, 0);
  } else if ([name isEqualToString:@"y"]) {
    matrix->matrix[0] = simd_make_float2(0, 0);
    matrix->matrix[1] = simd_make_float2(0, -1);
    matrix->matrix[2] = simd_make_float2(0, 1);
    matrix->matrix[3] = simd_make_float2(0, 0);
  } else if ([name isEqualToString:@"z"]) {
    matrix->matrix[0] = simd_make_float2(1, 0);
    matrix->matrix[1] = simd_make_float2(0, 0);
    matrix->matrix[2] = simd_make_float2(0, 0);
    matrix->matrix[3] = simd_make_float2(-1, 0);
  } else if ([name isEqualToString:@"h"]) {
    matrix->matrix[0] = simd_make_float2(r2, 0);
    matrix->matrix[1] = simd_make_float2(r2, 0);
    matrix->matrix[2] = simd_make_float2(r2, 0);
    matrix->matrix[3] = simd_make_float2(-r2, 0);
  }

  // Simple rotation gates implementation
  else if ([name isEqualToString:@"rx"]) {
    float theta = [params[0] floatValue];
    float c = cosf(theta / 2);
    float s = sinf(theta / 2);
    matrix->matrix[0] = simd_make_float2(c, 0);
    matrix->matrix[1] = simd_make_float2(0, -s);
    matrix->matrix[2] = simd_make_float2(0, -s);
    matrix->matrix[3] = simd_make_float2(c, 0);
  } else if ([name isEqualToString:@"ry"]) {
    float theta = [params[0] floatValue];
    float c = cosf(theta / 2);
    float s = sinf(theta / 2);
    matrix->matrix[0] = simd_make_float2(c, 0);
    matrix->matrix[1] = simd_make_float2(-s, 0);
    matrix->matrix[2] = simd_make_float2(s, 0);
    matrix->matrix[3] = simd_make_float2(c, 0);
  } else if ([name isEqualToString:@"rz"] || [name isEqualToString:@"p"]) {
    float theta = [params[0] floatValue];
    matrix->matrix[0] = simd_make_float2(cosf(-theta / 2), sinf(-theta / 2));
    matrix->matrix[1] = simd_make_float2(0, 0);
    matrix->matrix[2] = simd_make_float2(0, 0);
    matrix->matrix[3] = simd_make_float2(cosf(theta / 2), sinf(theta / 2));
  }
}

- (void)_getGateMatrix2Q:(NSString *)name
                  params:(NSArray *)params
                  result:(GateMatrix2Q *)matrix {

  if ([name isEqualToString:@"swap"]) {
    // Simple I for non-swap components, Swap the middle.
    // 00 -> 00
    // 01 -> 10
    // 10 -> 01
    // 11 -> 11
    memset(matrix, 0, sizeof(GateMatrix2Q));
    matrix->matrix[0] = simd_make_float2(1, 0);  // [0,0]
    matrix->matrix[6] = simd_make_float2(1, 0);  // [1,2] -> Row 1 Col 2
    matrix->matrix[9] = simd_make_float2(1, 0);  // [2,1] -> Row 2 Col 1
    matrix->matrix[15] = simd_make_float2(1, 0); // [3,3]
  }
}

- (BOOL)_isControlledGate:(NSString *)name {
  // Basic heuristic: starts with 'c' but isn't 'swap' related (cswap is 3q, not
  // handled here yet) or known 2q gates.
  if ([name isEqualToString:@"cx"] || [name isEqualToString:@"cz"] ||
      [name isEqualToString:@"ch"] || [name isEqualToString:@"cp"] ||
      [name isEqualToString:@"crx"] || [name isEqualToString:@"cry"] ||
      [name isEqualToString:@"crz"]) {
    return YES;
  }
  return NO;
}

- (NSString *)_baseGateName:(NSString *)name {
  // cx -> x, crz -> rz, etc.
  // Assumes single letter control prefix for now
  if ([name hasPrefix:@"c"]) {
    return [name substringFromIndex:1];
  }
  return name;
}

#pragma mark - GPU Dispatches

- (MetalQError)_apply1QGate:(GateMatrix1Q *)matrix
                     target:(int)target
                stateVector:(MetalQStateVector *)sv {

  id<MTLCommandBuffer> cmdBuffer = [_commandQueue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

  [encoder setComputePipelineState:_gate1QPipeline];
  [encoder setBuffer:sv.realBuffer offset:0 atIndex:0];
  [encoder setBuffer:sv.imagBuffer offset:0 atIndex:1];
  [encoder setBytes:matrix length:sizeof(GateMatrix1Q) atIndex:2];

  GateParams params = {.targetQubit = (uint32_t)target,
                       .controlQubit = UINT32_MAX,
                       .numQubits = (uint32_t)sv.numQubits,
                       .stateSize = (uint32_t)sv.size};
  [encoder setBytes:&params length:sizeof(GateParams) atIndex:3];

  NSUInteger threadCount = sv.size / 2;
  MTLSize gridSize = MTLSizeMake(threadCount, 1, 1);
  MTLSize threadGroupSize = MTLSizeMake(
      MIN(256, _gate1QPipeline.maxTotalThreadsPerThreadgroup), 1, 1);

  [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
  [encoder endEncoding];

  [cmdBuffer commit];
  [cmdBuffer waitUntilCompleted];

  if (cmdBuffer.error) {
    NSLog(@"GPU Error: %@", cmdBuffer.error);
    return METALQ_ERROR_GPU_ERROR;
  }
  return METALQ_SUCCESS;
}

- (MetalQError)_applyControlledGate:(GateMatrix1Q *)matrix
                            control:(int)control
                             target:(int)target
                        stateVector:(MetalQStateVector *)sv {

  id<MTLCommandBuffer> cmdBuffer = [_commandQueue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

  [encoder setComputePipelineState:_controlledGatePipeline];
  [encoder setBuffer:sv.realBuffer offset:0 atIndex:0];
  [encoder setBuffer:sv.imagBuffer offset:0 atIndex:1];
  [encoder setBytes:matrix length:sizeof(GateMatrix1Q) atIndex:2];

  GateParams params = {.targetQubit = (uint32_t)target,
                       .controlQubit = (uint32_t)control,
                       .numQubits = (uint32_t)sv.numQubits,
                       .stateSize = (uint32_t)sv.size};
  [encoder setBytes:&params length:sizeof(GateParams) atIndex:3];

  // Controlled gate works on half the states where control bit is 1.
  // Actually we iterate half the total states (pairs) but logic inside selects
  // appropriate ones? The shader logic given in spec processes indices where
  // control=1 and target=0/1 pairs. The grid should cover enough threads.
  // Shader iterates `thread_position_in_grid`.
  // Logic:
  //   base = construct index i where control=1, target=0
  //   total such indices = size / 4 (since 2 bits are fixed relative to each
  //   other)

  NSUInteger threadCount = sv.size / 4;
  MTLSize gridSize = MTLSizeMake(threadCount, 1, 1);
  MTLSize threadGroupSize = MTLSizeMake(
      MIN(256, _controlledGatePipeline.maxTotalThreadsPerThreadgroup), 1, 1);

  [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
  [encoder endEncoding];

  [cmdBuffer commit];
  [cmdBuffer waitUntilCompleted];

  return cmdBuffer.error ? METALQ_ERROR_GPU_ERROR : METALQ_SUCCESS;
}

- (MetalQError)_apply2QGate:(GateMatrix2Q *)matrix
                     qubit0:(int)qubit0
                     qubit1:(int)qubit1
                stateVector:(MetalQStateVector *)sv {

  id<MTLCommandBuffer> cmdBuffer = [_commandQueue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

  [encoder setComputePipelineState:_gate2QPipeline];
  [encoder setBuffer:sv.realBuffer offset:0 atIndex:0];
  [encoder setBuffer:sv.imagBuffer offset:0 atIndex:1];
  [encoder setBytes:matrix length:sizeof(GateMatrix2Q) atIndex:2];

  GateParams params = {.targetQubit = (uint32_t)qubit0,
                       .controlQubit = (uint32_t)qubit1,
                       .numQubits = (uint32_t)sv.numQubits,
                       .stateSize = (uint32_t)sv.size};
  [encoder setBytes:&params length:sizeof(GateParams) atIndex:3];

  // 2Q gate processes groups of 4 indices (00,01,10,11)
  // Total threads needed = size / 4
  NSUInteger threadCount = sv.size / 4;
  MTLSize gridSize = MTLSizeMake(threadCount, 1, 1);
  MTLSize threadGroupSize = MTLSizeMake(
      MIN(256, _gate2QPipeline.maxTotalThreadsPerThreadgroup), 1, 1);

  [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
  [encoder endEncoding];

  [cmdBuffer commit];
  [cmdBuffer waitUntilCompleted];

  return cmdBuffer.error ? METALQ_ERROR_GPU_ERROR : METALQ_SUCCESS;
}

@end
