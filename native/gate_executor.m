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
  simd_float2 matrix[64]; // 8x8 complex matrix
} GateMatrix3Q;

typedef struct {
  uint32_t targetQubit;
  uint32_t controlQubit; // for controlled gates
  uint32_t extraQubit;   // for 3Q gates
  uint32_t numQubits;
  uint32_t stateSize;
} GateParams;

@interface MetalQGateExecutor (Private)
- (BOOL)_createPipelines;
- (void)_getGateMatrix3Q:(NSString *)name
                  params:(NSArray *)params
                  result:(GateMatrix3Q *)matrix;
- (MetalQError)_apply3QGate:(GateMatrix3Q *)matrix
                     qubit0:(int)q0
                     qubit1:(int)q1
                     qubit2:(int)q2
                stateVector:(MetalQStateVector *)sv;
- (void)_parseMatrix1Q:(NSArray *)data result:(GateMatrix1Q *)matrix;
- (void)_parseMatrix2Q:(NSArray *)data result:(GateMatrix2Q *)matrix;
- (void)_parseMatrix3Q:(NSArray *)data result:(GateMatrix3Q *)matrix;
@end

@implementation MetalQGateExecutor {
  id<MTLDevice> _device;
  id<MTLCommandQueue> _commandQueue;
  id<MTLLibrary> _library;

  // Pipeline states
  id<MTLComputePipelineState> _gate1QPipeline;
  id<MTLComputePipelineState> _gate2QPipeline;
  id<MTLComputePipelineState> _gate3QPipeline;
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
        NSLog(@"Metal-Q Debug: Loaded library from %@", path);
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
  id<MTLFunction> gate3QFunc = [_library newFunctionWithName:@"apply_gate_3q"];
  id<MTLFunction> ctrlGateFunc =
      [_library newFunctionWithName:@"apply_controlled_gate"];

  if (!gate1QFunc || !gate2QFunc || !gate3QFunc || !ctrlGateFunc) {
    NSLog(@"Metal-Q: Failed to find shader functions");
    return NO;
  }

  _gate1QPipeline = [_device newComputePipelineStateWithFunction:gate1QFunc
                                                           error:&error];
  _gate2QPipeline = [_device newComputePipelineStateWithFunction:gate2QFunc
                                                           error:&error];
  _gate3QPipeline = [_device newComputePipelineStateWithFunction:gate3QFunc
                                                           error:&error];
  _controlledGatePipeline =
      [_device newComputePipelineStateWithFunction:ctrlGateFunc error:&error];

  if (!_gate1QPipeline || !_gate2QPipeline || !_gate3QPipeline ||
      !_controlledGatePipeline) {
    NSLog(@"Metal-Q: Failed to create compute pipelines: %@", error);
    return NO;
  }
  return YES;
}

- (MetalQError)applyGate:(NSDictionary *)gateData
           toStateVector:(MetalQStateVector *)stateVector {
  NSString *name = gateData[@"name"];
  NSArray *qubits = gateData[@"qubits"];
  NSArray *params = gateData[@"params"];
  id matrixObj = gateData[@"matrix"];
  // NSLog(@"Metal-Q Debug: gate=%@ matrixObj=%@ type=%@", name, matrixObj,
  // [matrixObj class]);
  NSArray *matrixData = (matrixObj != [NSNull null]) ? matrixObj : nil;

  if ([qubits count] == 1) {
    if ([self _isControlledGate:name]) {
      // 1-qubit controlled gate (e.g. MCX with 1 target? No, controlled gate
      // usually means 2+ qubits) "cx" is 2 qubits.
      // If we have a controlled gate with 1 qubit list, it's invalid/impossible
      // unless implicit?
      // Metal-Q assumes "cx" has 2 qubits in the list.
      return METALQ_ERROR_INVALID_CIRCUIT;
    } else {
      GateMatrix1Q matrix1Q;
      if (matrixData) {
        // Parse explicit matrix
        // NSLog(@"Metal-Q Debug: Parsing 1Q Matrix: %@", matrixData);
        [self _parseMatrix1Q:matrixData result:&matrix1Q];
      } else {
        [self _getGateMatrix1Q:name params:params result:&matrix1Q];
      }
      return [self _apply1QGate:&matrix1Q
                         target:[qubits[0] intValue]
                    stateVector:stateVector];
    }
  } else if ([qubits count] == 2) {
    if ([self _isControlledGate:name]) {
      // Controlled gate: q0 is control, q1 is target (Qiskit convention matches
      // this? cx(0, 1) -> 0 control, 1 target).
      GateMatrix1Q baseGate;
      NSString *baseName = [self _baseGateName:name];

      // Check if base gate is custom unitary?
      // Current architecture for controlled gates assumes standard base gates.
      // If "unitary" is passed, it is likely the full 2Q unitary, not a
      // controlled-unitary.
      // So custom unitaries will fall into the 'else' block below.

      [self _getGateMatrix1Q:baseName params:params result:&baseGate];

      // Debug Log (commented out for production)
      // NSLog(@"Metal-Q Debug: Controlled Gate %@ (base %@) c=%d t=%d", name,
      //       baseName, [qubits[0] intValue], [qubits[1] intValue]);
      // NSLog(@"Matrix: [%f, %f] [%f, %f]", baseGate.matrix[0].x,
      //       baseGate.matrix[0].y, baseGate.matrix[1].x,
      //       baseGate.matrix[1].y);
      // NSLog(@"        [%f, %f] [%f, %f]", baseGate.matrix[2].x,
      //       baseGate.matrix[2].y, baseGate.matrix[3].x,
      //       baseGate.matrix[3].y);

      return [self _applyControlledGate:&baseGate
                                control:[qubits[0] intValue]
                                 target:[qubits[1] intValue]
                            stateVector:stateVector];
    } else {
      GateMatrix2Q matrix2Q;
      if (matrixData) {
        [self _parseMatrix2Q:matrixData result:&matrix2Q];
      } else {
        [self _getGateMatrix2Q:name params:params result:&matrix2Q];
      }
      return [self _apply2QGate:&matrix2Q
                         qubit0:[qubits[0] intValue]
                         qubit1:[qubits[1] intValue]
                    stateVector:stateVector];
    }
  } else if ([qubits count] == 3) {
    GateMatrix3Q matrix3Q;
    if (matrixData) {
      [self _parseMatrix3Q:matrixData result:&matrix3Q];
    } else {
      [self _getGateMatrix3Q:name params:params result:&matrix3Q];
    }
    return [self _apply3QGate:&matrix3Q
                       qubit0:[qubits[0] intValue]
                       qubit1:[qubits[1] intValue]
                       qubit2:[qubits[2] intValue]
                  stateVector:stateVector];
  }

  return METALQ_ERROR_UNSUPPORTED_GATE;
}

#pragma mark - Matrix Parsing

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
  } else if ([name isEqualToString:@"s"]) {
    matrix->matrix[0] = simd_make_float2(1, 0);
    matrix->matrix[1] = simd_make_float2(0, 0);
    matrix->matrix[2] = simd_make_float2(0, 0);
    matrix->matrix[3] = simd_make_float2(0, 1);
  } else if ([name isEqualToString:@"sdg"]) {
    matrix->matrix[0] = simd_make_float2(1, 0);
    matrix->matrix[1] = simd_make_float2(0, 0);
    matrix->matrix[2] = simd_make_float2(0, 0);
    matrix->matrix[3] = simd_make_float2(0, -1);
  } else if ([name isEqualToString:@"t"]) {
    matrix->matrix[0] = simd_make_float2(1, 0);
    matrix->matrix[1] = simd_make_float2(0, 0);
    matrix->matrix[2] = simd_make_float2(0, 0);
    matrix->matrix[3] = simd_make_float2(r2, r2);
  } else if ([name isEqualToString:@"tdg"]) {
    matrix->matrix[0] = simd_make_float2(1, 0);
    matrix->matrix[1] = simd_make_float2(0, 0);
    matrix->matrix[2] = simd_make_float2(0, 0);
    matrix->matrix[3] = simd_make_float2(r2, -r2);
  } else if ([name isEqualToString:@"sx"]) {
    matrix->matrix[0] = simd_make_float2(0.5f, 0.5f);
    matrix->matrix[1] = simd_make_float2(0.5f, -0.5f);
    matrix->matrix[2] = simd_make_float2(0.5f, -0.5f);
    matrix->matrix[3] = simd_make_float2(0.5f, 0.5f);
  } else if ([name isEqualToString:@"sxdg"]) {
    matrix->matrix[0] = simd_make_float2(0.5f, -0.5f);
    matrix->matrix[1] = simd_make_float2(0.5f, 0.5f);
    matrix->matrix[2] = simd_make_float2(0.5f, 0.5f);
    matrix->matrix[3] = simd_make_float2(0.5f, -0.5f);
  }

  // Rotation and U gates
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
  } else if ([name isEqualToString:@"rz"] || [name isEqualToString:@"p"] ||
             [name isEqualToString:@"u1"]) {
    // Note: u1(lambda) == p(lambda)
    float theta = [params[0] floatValue];
    // RZ is exp(-i*theta/2*Z). P is diag(1, e^i*lambda). They differ by global
    // phase e^(-i*lambda/2). Metal-Q RZ implementation: diag(e^(-i*theta/2),
    // e^(i*theta/2)) P gate: diag(1, e^i*theta).

    if ([name isEqualToString:@"p"] || [name isEqualToString:@"u1"]) {
      matrix->matrix[0] = simd_make_float2(1, 0);
      matrix->matrix[1] = simd_make_float2(0, 0);
      matrix->matrix[2] = simd_make_float2(0, 0);
      matrix->matrix[3] = simd_make_float2(cosf(theta), sinf(theta));
    } else {
      matrix->matrix[0] = simd_make_float2(cosf(-theta / 2), sinf(-theta / 2));
      matrix->matrix[1] = simd_make_float2(0, 0);
      matrix->matrix[2] = simd_make_float2(0, 0);
      matrix->matrix[3] = simd_make_float2(cosf(theta / 2), sinf(theta / 2));
    }
  } else if ([name isEqualToString:@"u"] || [name isEqualToString:@"u3"]) {
    float theta = [params[0] floatValue];
    float phi = [params[1] floatValue];
    float lambda = [params[2] floatValue];
    float c = cosf(theta / 2);
    float s = sinf(theta / 2);
    matrix->matrix[0] = simd_make_float2(c, 0);
    matrix->matrix[1] = simd_make_float2(-s * cosf(lambda), -s * sinf(lambda));
    matrix->matrix[2] = simd_make_float2(s * cosf(phi), s * sinf(phi));
    matrix->matrix[3] =
        simd_make_float2(c * cosf(phi + lambda), c * sinf(phi + lambda));
  } else if ([name isEqualToString:@"u2"]) {
    float phi = [params[0] floatValue];
    float lambda = [params[1] floatValue];
    matrix->matrix[0] = simd_make_float2(r2, 0);
    matrix->matrix[1] =
        simd_make_float2(-r2 * cosf(lambda), -r2 * sinf(lambda));
    matrix->matrix[2] = simd_make_float2(r2 * cosf(phi), r2 * sinf(phi));
    matrix->matrix[3] =
        simd_make_float2(r2 * cosf(phi + lambda), r2 * sinf(phi + lambda));
  } else if ([name isEqualToString:@"r"]) {
    float theta = [params[0] floatValue];
    float phi = [params[1] floatValue];
    float c = cosf(theta / 2);
    float s = sinf(theta / 2);
    matrix->matrix[0] = simd_make_float2(c, 0);
    matrix->matrix[1] = simd_make_float2(-s * sinf(-phi), -s * cosf(-phi));
    matrix->matrix[2] = simd_make_float2(-s * sinf(phi), -s * cosf(phi));
    matrix->matrix[3] = simd_make_float2(c, 0);
  }
}

- (BOOL)_isControlledGate:(NSString *)name {
  // Broaden check for controlled gates in Qiskit 2.x
  NSSet *controlledGates = [NSSet setWithArray:@[
    @"cx", @"cy", @"cz", @"ch", @"cs", @"csdg", @"csx", @"cp", @"crx", @"cry",
    @"crz", @"cu", @"cu1", @"cu3"
  ]];
  return [controlledGates containsObject:name];
}

- (NSString *)_baseGateName:(NSString *)name {
  // Support broader range of controlled gates
  // Simple heuristic: if known controlled gate, strip prefix/map to base
  if ([name isEqualToString:@"cx"])
    return @"x";
  if ([name isEqualToString:@"cy"])
    return @"y";
  if ([name isEqualToString:@"cz"])
    return @"z";
  if ([name isEqualToString:@"ch"])
    return @"h";
  if ([name isEqualToString:@"cs"])
    return @"s";
  if ([name isEqualToString:@"csdg"])
    return @"sdg";
  if ([name isEqualToString:@"csx"])
    return @"sx";
  if ([name isEqualToString:@"cp"])
    return @"p";
  if ([name isEqualToString:@"crx"])
    return @"rx";
  if ([name isEqualToString:@"cry"])
    return @"ry";
  if ([name isEqualToString:@"crz"])
    return @"rz";
  if ([name isEqualToString:@"cu"])
    return @"u";
  if ([name isEqualToString:@"cu1"])
    return @"u1";
  if ([name isEqualToString:@"cu3"])
    return @"u3";

  if ([name hasPrefix:@"c"]) {
    return [name substringFromIndex:1];
  }
  return name;
}

#pragma mark - GPU Dispatches

- (void)_getGateMatrix2Q:(NSString *)name
                  params:(NSArray *)params
                  result:(GateMatrix2Q *)matrix {

  memset(matrix, 0, sizeof(GateMatrix2Q));

  if ([name isEqualToString:@"swap"]) {
    matrix->matrix[0] = simd_make_float2(1, 0);
    matrix->matrix[6] = simd_make_float2(1, 0);
    matrix->matrix[9] = simd_make_float2(1, 0);
    matrix->matrix[15] = simd_make_float2(1, 0);
  } else if ([name isEqualToString:@"iswap"]) {
    // iSWAP: |00>->|00>, |01>->i|10>, |10>->i|01>, |11>->|11>
    matrix->matrix[0] = simd_make_float2(1, 0);
    matrix->matrix[6] = simd_make_float2(0, 1);
    matrix->matrix[9] = simd_make_float2(0, 1);
    matrix->matrix[15] = simd_make_float2(1, 0);
  } else if ([name isEqualToString:@"dcx"]) {
    // DCX: 00->00, 01->10, 10->11, 11->01
    matrix->matrix[0] = simd_make_float2(1, 0);
    matrix->matrix[5] = simd_make_float2(1, 0);
    matrix->matrix[11] = simd_make_float2(1, 0);
    matrix->matrix[14] = simd_make_float2(1, 0);
  } else if ([name isEqualToString:@"ecr"]) {
    // ECR = (1/sqrt(2)) * [[0, 1, i, 0], [1, 0, 0, -i], [-i, 0, 0, 1], [0, i,
    // 1, 0]]
    float r2 = 1.0f / sqrtf(2.0f);
    matrix->matrix[1] = simd_make_float2(r2, 0);
    matrix->matrix[2] = simd_make_float2(0, r2);
    matrix->matrix[4] = simd_make_float2(r2, 0);
    matrix->matrix[7] = simd_make_float2(0, -r2);
    matrix->matrix[8] = simd_make_float2(0, -r2);
    matrix->matrix[11] = simd_make_float2(r2, 0);
    matrix->matrix[13] = simd_make_float2(0, r2);
    matrix->matrix[14] = simd_make_float2(r2, 0);
  } else if ([name isEqualToString:@"rxx"]) {
    float theta = [params[0] floatValue];
    float c = cosf(theta / 2);
    float s = sinf(theta / 2);
    // RXX = exp(-i*theta/2*X⊗X)
    matrix->matrix[0] = simd_make_float2(c, 0);
    matrix->matrix[3] = simd_make_float2(0, -s);
    matrix->matrix[5] = simd_make_float2(c, 0);
    matrix->matrix[6] = simd_make_float2(0, -s);
    matrix->matrix[9] = simd_make_float2(0, -s);
    matrix->matrix[10] = simd_make_float2(c, 0);
    matrix->matrix[12] = simd_make_float2(0, -s);
    matrix->matrix[15] = simd_make_float2(c, 0);
  } else if ([name isEqualToString:@"ryy"]) {
    float theta = [params[0] floatValue];
    float c = cosf(theta / 2);
    float s = sinf(theta / 2);
    // RYY = exp(-i*theta/2*Y⊗Y)
    matrix->matrix[0] = simd_make_float2(c, 0);
    matrix->matrix[3] = simd_make_float2(0, s);
    matrix->matrix[5] = simd_make_float2(c, 0);
    matrix->matrix[6] = simd_make_float2(0, -s);
    matrix->matrix[9] = simd_make_float2(0, -s);
    matrix->matrix[10] = simd_make_float2(c, 0);
    matrix->matrix[12] = simd_make_float2(0, s);
    matrix->matrix[15] = simd_make_float2(c, 0);
  } else if ([name isEqualToString:@"rzz"]) {
    float theta = [params[0] floatValue];
    float c = cosf(theta / 2);
    float s = sinf(theta / 2);
    // RZZ = exp(-i*theta/2*Z⊗Z) = diag(e^-i, e^i, e^i, e^-i)
    // Actually it is [c, -is] diagonal? No.
    // exp(-i*theta/2 * Z⊗Z)
    // 00 -> Z⊗Z = 1  -> exp(-i*theta/2)
    // 01 -> Z⊗Z = -1 -> exp(i*theta/2)
    // 10 -> Z⊗Z = -1 -> exp(i*theta/2)
    // 11 -> Z⊗Z = 1  -> exp(-i*theta/2)

    matrix->matrix[0] = simd_make_float2(cosf(-theta / 2), sinf(-theta / 2));
    matrix->matrix[5] = simd_make_float2(cosf(theta / 2), sinf(theta / 2));
    matrix->matrix[10] = simd_make_float2(cosf(theta / 2), sinf(theta / 2));
    matrix->matrix[15] = simd_make_float2(cosf(-theta / 2), sinf(-theta / 2));
  } else if ([name isEqualToString:@"rzx"]) {
    float theta = [params[0] floatValue];
    float c = cosf(theta / 2);
    float s = sinf(theta / 2);
    // RZX = exp(-i*theta/2*Z⊗X)
    // I use X⊗Z convention? Qiskit is Z_1 \otimes X_0 (qubits 1 and 0).
    // If native is little-endian (q0 lower bits), tensor product is usually (q1
    // q0). Check convention: gate matrix is 4x4, usually ordered 00, 01, 10, 11
    // (q1 q0 i.e. MSB LSB). Qiskit doc: RZX(theta) = exp(-i theta/2 Z \otimes
    // X) where Z is on q1 (control-like), X on q0 (target-like). If 4x4 matrix
    // is standard basis:
    //  RZX = [[c, -is, 0, 0], [-is, c, 0, 0], [0, 0, c, is], [0, 0, is, c]]

    matrix->matrix[0] = simd_make_float2(c, 0);
    matrix->matrix[1] = simd_make_float2(0, -s);
    matrix->matrix[4] = simd_make_float2(0, -s);
    matrix->matrix[5] = simd_make_float2(c, 0);

    matrix->matrix[10] = simd_make_float2(c, 0);
    matrix->matrix[11] = simd_make_float2(0, s);
    matrix->matrix[14] = simd_make_float2(0, s);
    matrix->matrix[15] = simd_make_float2(c, 0);
  }
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

- (MetalQError)_apply3QGate:(GateMatrix3Q *)matrix
                     qubit0:(int)q0
                     qubit1:(int)q1
                     qubit2:(int)q2
                stateVector:(MetalQStateVector *)sv {

  id<MTLCommandBuffer> cmdBuffer = [_commandQueue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

  [encoder setComputePipelineState:_gate3QPipeline];
  [encoder setBuffer:sv.realBuffer offset:0 atIndex:0];
  [encoder setBuffer:sv.imagBuffer offset:0 atIndex:1];
  [encoder setBytes:matrix length:sizeof(GateMatrix3Q) atIndex:2];

  GateParams params = {.targetQubit = (uint32_t)q0,
                       .controlQubit = (uint32_t)q1,
                       .extraQubit = (uint32_t)q2,
                       .numQubits = (uint32_t)sv.numQubits,
                       .stateSize = (uint32_t)sv.size};
  [encoder setBytes:&params length:sizeof(GateParams) atIndex:3];

  NSUInteger threadCount = sv.size / 8;
  MTLSize gridSize = MTLSizeMake(threadCount, 1, 1);
  MTLSize threadGroupSize = MTLSizeMake(
      MIN(256, _gate3QPipeline.maxTotalThreadsPerThreadgroup), 1, 1);

  [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
  [encoder endEncoding];

  [cmdBuffer commit];
  [cmdBuffer waitUntilCompleted];

  return cmdBuffer.error ? METALQ_ERROR_GPU_ERROR : METALQ_SUCCESS;
}

- (void)_getGateMatrix3Q:(NSString *)name
                  params:(NSArray *)params
                  result:(GateMatrix3Q *)matrix {
  memset(matrix, 0, sizeof(GateMatrix3Q));

  // Diagonal Identity default for all? no, just helpers.
  // Helper to set Identity diagonal
  for (int i = 0; i < 8; i++)
    matrix->matrix[i * 8 + i] = simd_make_float2(1, 0);

  if ([name isEqualToString:@"ccx"]) {
    // Swap 3 (011 - c1,c2=1, t=0) and 7 (111 - c1,c2=1, t=1)
    matrix->matrix[3 * 8 + 3] = simd_make_float2(0, 0);
    matrix->matrix[7 * 8 + 7] = simd_make_float2(0, 0);

    matrix->matrix[3 * 8 + 7] = simd_make_float2(1, 0);
    matrix->matrix[7 * 8 + 3] = simd_make_float2(1, 0);
  } else if ([name isEqualToString:@"cswap"]) {
    // Swap 3 (011 - c=1, t1=1, t2=0) and 5 (101 - c=1, t1=0, t2=1)
    matrix->matrix[3 * 8 + 3] = simd_make_float2(0, 0);
    matrix->matrix[5 * 8 + 5] = simd_make_float2(0, 0);

    matrix->matrix[3 * 8 + 5] = simd_make_float2(1, 0);
    matrix->matrix[5 * 8 + 3] = simd_make_float2(1, 0);
  } else if ([name isEqualToString:@"ccz"]) {
    matrix->matrix[7 * 8 + 7] = simd_make_float2(-1, 0);
  }
}

- (void)_parseMatrix1Q:(NSArray *)data result:(GateMatrix1Q *)matrix {
  for (int r = 0; r < 2; r++) {
    NSArray *row = data[r];
    for (int c = 0; c < 2; c++) {
      NSArray *val = row[c];
      float re = [val[0] floatValue];
      float im = [val[1] floatValue];
      matrix->matrix[r * 2 + c] = simd_make_float2(re, im);
    }
  }
}

- (void)_parseMatrix2Q:(NSArray *)data result:(GateMatrix2Q *)matrix {
  for (int r = 0; r < 4; r++) {
    NSArray *row = data[r];
    for (int c = 0; c < 4; c++) {
      NSArray *val = row[c];
      float re = [val[0] floatValue];
      float im = [val[1] floatValue];
      matrix->matrix[r * 4 + c] = simd_make_float2(re, im);
    }
  }
}

- (void)_parseMatrix3Q:(NSArray *)data result:(GateMatrix3Q *)matrix {
  for (int r = 0; r < 8; r++) {
    NSArray *row = data[r];
    for (int c = 0; c < 8; c++) {
      NSArray *val = row[c];
      float re = [val[0] floatValue];
      float im = [val[1] floatValue];
      matrix->matrix[r * 8 + c] = simd_make_float2(re, im);
    }
  }
}
@end
