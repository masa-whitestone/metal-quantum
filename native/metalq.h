/**
 * metalq.h - Metal-Q Public C API
 * 
 * Quantum circuit simulator using Apple Metal GPU
 */

#ifndef METALQ_H
#define METALQ_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Error codes
 */
typedef enum {
    METALQ_SUCCESS = 0,
    METALQ_ERROR_INIT_FAILED = -1,
    METALQ_ERROR_INVALID_CIRCUIT = -2,
    METALQ_ERROR_INVALID_SHOTS = -3,
    METALQ_ERROR_GPU_ERROR = -4,
    METALQ_ERROR_OUT_OF_MEMORY = -5,
    METALQ_ERROR_UNSUPPORTED_GATE = -6,
} MetalQError;

/**
 * Initialize the library
 * 
 * @return METALQ_SUCCESS on success, error code otherwise
 */
int metalq_init(void);

/**
 * Run a quantum circuit
 * 
 * @param circuit_json  Circuit data (JSON string)
 * @param shots         Number of shots (1-8192)
 * @param result_json   Pointer to result JSON string (output)
 * @param result_length Length of result string (output)
 * @return METALQ_SUCCESS on success, error code otherwise
 * 
 * Caller must free the result using metalq_free_result
 */
int metalq_run_circuit(
    const char* circuit_json,
    int shots,
    char** result_json,
    int* result_length
);

/**
 * Get statevector
 * 
 * @param circuit_json  Circuit data (JSON string, no measurements)
 * @param real_parts    Pointer to real parts array (output)
 * @param imag_parts    Pointer to imaginary parts array (output)
 * @param length        Array length (output)
 * @return METALQ_SUCCESS on success, error code otherwise
 * 
 * Caller must free the arrays using metalq_free_statevector
 */
int metalq_get_statevector(
    const char* circuit_json,
    float** real_parts,
    float** imag_parts,
    int* length
);

/**
 * Free result memory
 */
void metalq_free_result(char* result);

/**
 * Free statevector memory
 */
void metalq_free_statevector(float* real, float* imag);

/**
 * Cleanup library resources
 */
void metalq_cleanup(void);

/**
 * Get version string
 */
const char* metalq_version(void);

#ifdef __cplusplus
}
#endif

#endif /* METALQ_H */
