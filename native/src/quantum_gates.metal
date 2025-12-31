/*
 * quantum_gates.metal - Quantum Gate Kernels
 *
 * Implements standard quantum gates for statevector simulation.
 */

#include <metal_stdlib>
using namespace metal;

// Complex number structure
struct Complex {
    float real;
    float imag;
};

// Free functions for complex arithmetic to handle address space implicitly via value copying
Complex c_add(Complex a, Complex b) {
    return {a.real + b.real, a.imag + b.imag};
}

Complex c_sub(Complex a, Complex b) {
    return {a.real - b.real, a.imag - b.imag};
}

Complex c_mul(Complex a, Complex b) {
    return {a.real * b.real - a.imag * b.imag,
            a.real * b.imag + a.imag * b.real};
}

// ===========================================================================
// Single Qubit Gates
// ===========================================================================

struct GateMatrix2x2 {
    Complex val[4];
};

kernel void apply_gate_single(
    device Complex* state [[buffer(0)]],
    constant uint& target_bit [[buffer(1)]],
    constant GateMatrix2x2& gate [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint step = 1 << target_bit;
    
    // Calculate indices
    uint idx0 = (id / step) * (step << 1) + (id % step);
    uint idx1 = idx0 + step;
    
    // Load state (device -> thread)
    Complex a0 = state[idx0];
    Complex a1 = state[idx1];
    
    // Load gate matrix (constant -> thread)
    Complex g00 = gate.val[0];
    Complex g01 = gate.val[1];
    Complex g10 = gate.val[2];
    Complex g11 = gate.val[3];
    
    // |0'> = G00|0> + G01|1>
    // |1'> = G10|0> + G11|1>
    
    Complex term00 = c_mul(g00, a0);
    Complex term01 = c_mul(g01, a1);
    Complex b0 = c_add(term00, term01);
    
    Complex term10 = c_mul(g10, a0);
    Complex term11 = c_mul(g11, a1);
    Complex b1 = c_add(term10, term11);
    
    state[idx0] = b0;
    state[idx1] = b1;
}

// ===========================================================================
// Two Qubit Gates
// ===========================================================================

struct GateMatrix4x4 {
    Complex val[16];
};

kernel void apply_gate_two(
    device Complex* state [[buffer(0)]],
    constant uint& control_bit [[buffer(1)]],
    constant uint& target_bit [[buffer(2)]],
    constant GateMatrix4x4& gate [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint pos1 = min(control_bit, target_bit);
    uint pos2 = max(control_bit, target_bit);
    
    uint step1 = 1 << pos1;
    uint step2 = 1 << pos2;
    
    uint base = id;
    
    // Insert 0 at pos1
    base = (base / step1) * (step1 << 1) + (base % step1);
    // Insert 0 at pos2
    base = (base / step2) * (step2 << 1) + (base % step2);
    
    uint i_c = 1 << control_bit;
    uint i_t = 1 << target_bit;
    
    uint indices[4];
    indices[0] = base;             // 00
    indices[1] = base + i_t;       // 01 
    indices[2] = base + i_c;       // 10 
    indices[3] = base + i_c + i_t; // 11
    
    // Load values
    Complex v[4];
    v[0] = state[indices[0]];
    v[1] = state[indices[1]];
    v[2] = state[indices[2]];
    v[3] = state[indices[3]];
    
    Complex res[4];
    
    // Manual unroll or loop with cached matrix values
    for (int r = 0; r < 4; r++) {
        res[r] = {0, 0};
        for (int c = 0; c < 4; c++) {
            Complex g_val = gate.val[r * 4 + c]; // constant -> thread
            Complex prod = c_mul(g_val, v[c]);
            res[r] = c_add(res[r], prod);
        }
    }
    
    state[indices[0]] = res[0];
    state[indices[1]] = res[1];
    state[indices[2]] = res[2];
    state[indices[3]] = res[3];
}
