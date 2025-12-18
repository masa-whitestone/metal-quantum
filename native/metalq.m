/**
 * metalq.m - Metal-Q Main Implementation
 */

#import "metalq.h"
#import "gate_executor.h"
#import "measurement.h"
#import "state_vector.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Global state
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_commandQueue = nil;
static MetalQGateExecutor *g_gateExecutor = nil;
static BOOL g_initialized = NO;

#pragma mark - Initialization

int metalq_init(void) {
  if (g_initialized) {
    return METALQ_SUCCESS;
  }

  @autoreleasepool {
    // Get default Metal device
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
      NSLog(@"Metal-Q: Failed to create Metal device");
      return METALQ_ERROR_INIT_FAILED;
    }

    // Create command queue
    g_commandQueue = [g_device newCommandQueue];
    if (!g_commandQueue) {
      NSLog(@"Metal-Q: Failed to create command queue");
      return METALQ_ERROR_INIT_FAILED;
    }

    // Initialize gate executor
    g_gateExecutor = [[MetalQGateExecutor alloc] initWithDevice:g_device
                                                   commandQueue:g_commandQueue];
    if (!g_gateExecutor) {
      NSLog(@"Metal-Q: Failed to initialize gate executor");
      return METALQ_ERROR_INIT_FAILED;
    }

    g_initialized = YES;
    // NSLog(@"Metal-Q: Initialized successfully on %@", g_device.name);
  }

  return METALQ_SUCCESS;
}

#pragma mark - Circuit Execution

int metalq_run_circuit(const char *circuit_json, int shots, char **result_json,
                       int *result_length) {
  if (!g_initialized) {
    return METALQ_ERROR_INIT_FAILED;
  }

  if (shots < 1 || shots > 8192) {
    return METALQ_ERROR_INVALID_SHOTS;
  }

  @autoreleasepool {
    // Parse JSON
    NSString *jsonString = [NSString stringWithUTF8String:circuit_json];
    NSError *error = nil;
    NSDictionary *circuitData = [NSJSONSerialization
        JSONObjectWithData:[jsonString dataUsingEncoding:NSUTF8StringEncoding]
                   options:0
                     error:&error];

    if (error) {
      NSLog(@"Metal-Q: JSON parse error: %@", error);
      return METALQ_ERROR_INVALID_CIRCUIT;
    }

    // Extract parameters
    int numQubits = [circuitData[@"num_qubits"] intValue];
    int numClbits = [circuitData[@"num_clbits"] intValue];
    NSArray *gates = circuitData[@"gates"];
    NSArray *measurements = circuitData[@"measurements"];

    if (numQubits < 1 || numQubits > 30) {
      NSLog(@"Metal-Q: Invalid qubit count: %d", numQubits);
      return METALQ_ERROR_INVALID_CIRCUIT;
    }

    // Initialize state vector
    MetalQStateVector *stateVector =
        [[MetalQStateVector alloc] initWithDevice:g_device numQubits:numQubits];

    if (!stateVector) {
      return METALQ_ERROR_OUT_OF_MEMORY;
    }

    // Apply gates
    for (NSDictionary *gateData in gates) {
      MetalQError err = [g_gateExecutor applyGate:gateData
                                    toStateVector:stateVector];
      if (err != METALQ_SUCCESS) {
        return err;
      }
    }

    // Measurement
    NSMutableDictionary *counts = [NSMutableDictionary dictionary];
    MetalQError err = [MetalQMeasurement sampleFromStateVector:stateVector
                                                  measurements:measurements
                                                     numClbits:numClbits
                                                         shots:shots
                                                       results:counts];

    if (err != METALQ_SUCCESS) {
      return err;
    }

    // Serialize result
    NSDictionary *resultDict = @{
      @"counts" : counts,
      @"shots" : @(shots),
      @"num_qubits" : @(numQubits),
      @"num_clbits" : @(numClbits)
    };

    NSData *resultData = [NSJSONSerialization dataWithJSONObject:resultDict
                                                         options:0
                                                           error:&error];

    if (error) {
      NSLog(@"Metal-Q: Result serialization error: %@", error);
      return METALQ_ERROR_GPU_ERROR;
    }

    // Allocate result string
    NSString *resultString =
        [[NSString alloc] initWithData:resultData
                              encoding:NSUTF8StringEncoding];

    *result_length =
        (int)[resultString lengthOfBytesUsingEncoding:NSUTF8StringEncoding];
    *result_json = (char *)malloc(*result_length + 1);
    strcpy(*result_json, [resultString UTF8String]);
  }

  return METALQ_SUCCESS;
}

#pragma mark - Statevector

int metalq_get_statevector(const char *circuit_json, float **real_parts,
                           float **imag_parts, int *length) {
  if (!g_initialized) {
    return METALQ_ERROR_INIT_FAILED;
  }

  @autoreleasepool {
    // Parse JSON
    NSString *jsonString = [NSString stringWithUTF8String:circuit_json];
    NSError *error = nil;
    NSDictionary *circuitData = [NSJSONSerialization
        JSONObjectWithData:[jsonString dataUsingEncoding:NSUTF8StringEncoding]
                   options:0
                     error:&error];

    if (error) {
      return METALQ_ERROR_INVALID_CIRCUIT;
    }

    int numQubits = [circuitData[@"num_qubits"] intValue];
    NSArray *gates = circuitData[@"gates"];
    // Ignore measurements for statevector calculation if any were passed,
    // but typically circuit_data should just have gates.

    // Initialize and Run
    MetalQStateVector *stateVector =
        [[MetalQStateVector alloc] initWithDevice:g_device numQubits:numQubits];

    if (!stateVector) {
      return METALQ_ERROR_OUT_OF_MEMORY;
    }

    for (NSDictionary *gateData in gates) {
      MetalQError err = [g_gateExecutor applyGate:gateData
                                    toStateVector:stateVector];
      if (err != METALQ_SUCCESS) {
        return err;
      }
    }

    // Copy to host
    *length = (int)stateVector.size;
    *real_parts = (float *)malloc(*length * sizeof(float));
    *imag_parts = (float *)malloc(*length * sizeof(float));

    [stateVector copyToHost:*real_parts imaginary:*imag_parts];
  }

  return METALQ_SUCCESS;
}

#pragma mark - Memory Management

void metalq_free_result(char *result) {
  if (result) {
    free(result);
  }
}

void metalq_free_statevector(float *real, float *imag) {
  if (real)
    free(real);
  if (imag)
    free(imag);
}

void metalq_cleanup(void) {
  @autoreleasepool {
    g_gateExecutor = nil;
    g_commandQueue = nil;
    g_device = nil;
    g_initialized = NO;
  }
}

const char *metalq_version(void) { return "0.1.0"; }
