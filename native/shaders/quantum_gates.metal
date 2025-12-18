/**
 * quantum_gates.metal - Metal Compute Shaders for Quantum Gates
 */

#include <metal_stdlib>
using namespace metal;

// Complex number operations
struct Complex {
    float real;
    float imag;
    
    Complex() : real(0), imag(0) {}
    Complex(float r, float i) : real(r), imag(i) {}
};

Complex cmul(Complex a, Complex b) {
    return Complex(
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    );
}

Complex cadd(Complex a, Complex b) {
    return Complex(a.real + b.real, a.imag + b.imag);
}

// Gate Matrix Structures
struct GateMatrix1Q {
    float2 matrix[4];  // 2x2 matrix, each element is (real, imag)
};

struct GateMatrix2Q {
    float2 matrix[16]; // 4x4 matrix
};

struct GateParams {
    uint targetQubit;
    uint controlQubit;
    uint numQubits;
    uint stateSize;
};

/**
 * Apply 1-Qubit Gate
 * 
 * Each thread handles a pair of amplitudes (amp[i], amp[j])
 * where j = i XOR (1 << targetQubit).
 * We invoke N/2 threads.
 */
kernel void apply_gate_1q(
    device float* real [[buffer(0)]],
    device float* imag [[buffer(1)]],
    constant GateMatrix1Q& gate [[buffer(2)]],
    constant GateParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint targetBit = 1u << params.targetQubit;
    
    // Construct index 'i' which has 0 at targetBit position
    // gid maps to the index among the N/2 pairs.
    // Insert a 0 bit at targetQubit position into gid.
    uint lowMask = targetBit - 1;
    uint highMask = ~lowMask;
    
    uint i = (gid & lowMask) | ((gid & (highMask >> 1)) << 1);
    uint j = i | targetBit;
    
    if (j >= params.stateSize) return;
    
    // Load amplitudes
    Complex amp_i = Complex(real[i], imag[i]);
    Complex amp_j = Complex(real[j], imag[j]);
    
    // Matrix elements (row-major)
    Complex m00 = Complex(gate.matrix[0].x, gate.matrix[0].y);
    Complex m01 = Complex(gate.matrix[1].x, gate.matrix[1].y);
    Complex m10 = Complex(gate.matrix[2].x, gate.matrix[2].y);
    Complex m11 = Complex(gate.matrix[3].x, gate.matrix[3].y);
    
    // Apply matrix: [new_i, new_j]^T = M * [amp_i, amp_j]^T
    Complex new_i = cadd(cmul(m00, amp_i), cmul(m01, amp_j));
    Complex new_j = cadd(cmul(m10, amp_i), cmul(m11, amp_j));
    
    // Store results
    real[i] = new_i.real;
    imag[i] = new_i.imag;
    real[j] = new_j.real;
    imag[j] = new_j.imag;
}

/**
 * Apply Controlled Gate (CX, CZ, etc.)
 * 
 * Only apply gate to target if control bit is 1.
 * We invoke N/4 threads (handling the subspace where control=1).
 */
kernel void apply_controlled_gate(
    device float* real [[buffer(0)]],
    device float* imag [[buffer(1)]],
    constant GateMatrix1Q& gate [[buffer(2)]],
    constant GateParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint controlBit = 1u << params.controlQubit;
    uint targetBit = 1u << params.targetQubit;
    
    // We need to construct indices where control=1.
    // AND target=0 (for i) / target=1 (for j).
    // So we invoke threads for N/4 pairs.
    // We insert two bits into gid: control (fixed to 1) and target (fixed to 0).
    
    uint minBit = min(params.controlQubit, params.targetQubit);
    uint maxBit = max(params.controlQubit, params.targetQubit);
    
    uint lowMask = (1u << minBit) - 1;
    uint midMask = ((1u << maxBit) - 1) ^ ((1u << (minBit + 1)) - 1);
    uint highMask = ~((1u << (maxBit + 1)) - 1);
    
    // Assemble base index: insert 0 at minBit and 0 at maxBit
    uint base = (gid & lowMask) 
              | ((gid << 1) & midMask)
              | ((gid << 2) & highMask);
    
    uint i = base | controlBit;           // control=1, target=0
    uint j = i | targetBit;               // control=1, target=1
    
    if (j >= params.stateSize) return;
    
    Complex amp_i = Complex(real[i], imag[i]);
    Complex amp_j = Complex(real[j], imag[j]);
    
    Complex m00 = Complex(gate.matrix[0].x, gate.matrix[0].y);
    Complex m01 = Complex(gate.matrix[1].x, gate.matrix[1].y);
    Complex m10 = Complex(gate.matrix[2].x, gate.matrix[2].y);
    Complex m11 = Complex(gate.matrix[3].x, gate.matrix[3].y);
    
    Complex new_i = cadd(cmul(m00, amp_i), cmul(m01, amp_j));
    Complex new_j = cadd(cmul(m10, amp_i), cmul(m11, amp_j));
    
    real[i] = new_i.real;
    imag[i] = new_i.imag;
    real[j] = new_j.real;
    imag[j] = new_j.imag;
}

/**
 * Apply General 2-Qubit Gate (SWAP, etc.)
 */
kernel void apply_gate_2q(
    device float* real [[buffer(0)]],
    device float* imag [[buffer(1)]],
    constant GateMatrix2Q& gate [[buffer(2)]],
    constant GateParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint bit0 = 1u << params.targetQubit;   
    uint bit1 = 1u << params.controlQubit;  
    
    uint minBit = min(params.targetQubit, params.controlQubit);
    uint maxBit = max(params.targetQubit, params.controlQubit);
    
    uint lowMask = (1u << minBit) - 1;
    uint midMask = ((1u << maxBit) - 1) ^ ((1u << (minBit + 1)) - 1);
    uint highMask = ~((1u << (maxBit + 1)) - 1);
    
    // Assemble base index: insert 0 at minBit and 0 at maxBit
    uint base = (gid & lowMask) 
              | ((gid << 1) & midMask)
              | ((gid << 2) & highMask);
    
    // 4 indices for 00, 01, 10, 11
    uint idx[4];
    idx[0] = base;                    
    idx[1] = base | bit0;             
    idx[2] = base | bit1;             
    idx[3] = base | bit0 | bit1;      
    
    if (idx[3] >= params.stateSize) return;
    
    Complex amp[4];
    for (int k = 0; k < 4; k++) {
        amp[k] = Complex(real[idx[k]], imag[idx[k]]);
    }
    
    Complex result[4];
    for (int row = 0; row < 4; row++) {
        result[row] = Complex(0, 0);
        for (int col = 0; col < 4; col++) {
            Complex m = Complex(
                gate.matrix[row * 4 + col].x,
                gate.matrix[row * 4 + col].y
            );
            result[row] = cadd(result[row], cmul(m, amp[col]));
        }
    }
    
    for (int k = 0; k < 4; k++) {
        real[idx[k]] = result[k].real;
        imag[idx[k]] = result[k].imag;
    }
}
