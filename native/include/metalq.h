/*
 * metalq.h - Metal-Q C API Header
 * Copyright (c) 2025 Masaki Shiraishi
 *
 * This header defines the C interface for the Metal-Q native library.
 * It is used by the Python ctypes/CFFI wrapper.
 */

#ifndef METALQ_H
#define METALQ_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===========================================================================
// Types and Handles
// ===========================================================================

typedef void *mq_context_t;
typedef void *mq_circuit_t;

typedef struct {
  uint32_t num_terms;
  uint32_t num_qubits;
  double *coeffs;       // [num_terms]
  uint8_t *pauli_codes; // [num_terms * num_qubits] (0=I, 1=X, 2=Y, 3=Z)
} mq_hamiltonian_t;

typedef struct {
  float real;
  float imag;
} mq_complex_t;

// ===========================================================================
// Context Management
// ===========================================================================

/**
 * Check if Metal is supported on this device.
 */
bool metalq_is_supported(void);

/**
 * Create a new Metal-Q context (initializes Metal device and command queue).
 * Returns NULL on failure.
 */
mq_context_t metalq_create_context(void);

/**
 * Destroy a Metal-Q context.
 */
void metalq_destroy_context(mq_context_t ctx);

// ===========================================================================
// Circuit Execution
// ===========================================================================

// Gate types
typedef enum {
  MQ_GATE_X = 0,
  MQ_GATE_Y,
  MQ_GATE_Z,
  MQ_GATE_H,
  MQ_GATE_S,
  MQ_GATE_T,
  MQ_GATE_RX,
  MQ_GATE_RY,
  MQ_GATE_RZ,
  MQ_GATE_P,
  MQ_GATE_CX,
  MQ_GATE_CY,
  MQ_GATE_CZ,
  MQ_GATE_SWAP,
  MQ_GATE_CP,
  MQ_GATE_U1,
  // ... add more as needed
} mq_gate_type_t;

typedef struct {
  mq_gate_type_t type;
  uint32_t qubits[3];
  uint32_t num_qubits;
  double params[3];
  uint32_t num_params;
} mq_gate_t;

/**
 * Execute a batch of gates on the statevector.
 *
 * @param ctx Context handle
 * @param num_qubits Number of qubits (defines statevector size 2^n)
 * @param gates Array of gates
 * @param num_gates Number of gates
 * @param shots Number of measurement shots (0 = statevector only)
 * @param out_statevector Buffer for output statevector (optional, size 2^n * 16
 * bytes)
 * @param out_counts Buffer for measurement counts (optional)
 *
 * Returns 0 on success, non-zero on error.
 */
int metalq_run(mq_context_t ctx, uint32_t num_qubits, const mq_gate_t *gates,
               uint32_t num_gates, uint32_t shots,
               mq_complex_t *out_statevector, uint64_t *out_counts);

// ===========================================================================
// Advanced Features
// ===========================================================================

/**
 * Compute gradient using Adjoint Differentiation.
 */
int metalq_gradient_adjoint(mq_context_t ctx, uint32_t num_qubits,
                            const mq_gate_t *gates, uint32_t num_gates,
                            void *hamiltonian, double *out_gradients);

#ifdef __cplusplus
}
#endif

#endif // METALQ_H
