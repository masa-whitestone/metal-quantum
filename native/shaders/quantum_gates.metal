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

typedef struct {
  float2 matrix[16]; // 4x4 complex matrix
} GateMatrix2Q;

typedef struct {
  float2 matrix[64]; // 8x8 complex matrix
} GateMatrix3Q;

typedef struct {
  uint targetQubit;
  uint controlQubit; // for controlled gates
  uint extraQubit;   // for 3Q gates
  uint numQubits;
  uint stateSize;
} GateParams;

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
    
    uint i = (gid & lowMask) | ((gid & highMask) << 1);
    uint j = i | targetBit;
    
    if (j >= params.stateSize) return;
    
    // Load amplitudes
    float amp_i_r = real[i]; float amp_i_i = imag[i];
    float amp_j_r = real[j]; float amp_j_i = imag[j];
    
    // Matrix elements (row-major)
    float m00_r = gate.matrix[0].x; float m00_i = gate.matrix[0].y;
    float m01_r = gate.matrix[1].x; float m01_i = gate.matrix[1].y;
    float m10_r = gate.matrix[2].x; float m10_i = gate.matrix[2].y;
    float m11_r = gate.matrix[3].x; float m11_i = gate.matrix[3].y;
    
    // Apply matrix: [new_i, new_j]^T = M * [amp_i, amp_j]^T
    // new_i = m00*amp_i + m01*amp_j
    float new_i_r = (m00_r * amp_i_r - m00_i * amp_i_i) + (m01_r * amp_j_r - m01_i * amp_j_i);
    float new_i_i = (m00_r * amp_i_i + m00_i * amp_i_r) + (m01_r * amp_j_i + m01_i * amp_j_r);
    
    // new_j = m10*amp_i + m11*amp_j
    float new_j_r = (m10_r * amp_i_r - m10_i * amp_i_i) + (m11_r * amp_j_r - m11_i * amp_j_i);
    float new_j_i = (m10_r * amp_i_i + m10_i * amp_i_r) + (m11_r * amp_j_i + m11_i * amp_j_r);
    
    // Store results
    real[i] = new_i_r;
    imag[i] = new_i_i;
    real[j] = new_j_r;
    imag[j] = new_j_i;
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
    
    // Simplified insertion logic
    uint minBit = min(params.controlQubit, params.targetQubit);
    uint maxBit = max(params.controlQubit, params.targetQubit);
    
    uint idx = gid;
    
    // Insert 0 at minBit
    uint maskMin = (1u << minBit) - 1;
    idx = ((idx & ~maskMin) << 1) | (idx & maskMin);
    
    // Insert 0 at maxBit
    uint maskMax = (1u << maxBit) - 1;
    idx = ((idx & ~maskMax) << 1) | (idx & maskMax);
    
    uint i = idx | controlBit;            // control=1, target=0
    uint j = i | targetBit;               // control=1, target=1
    
    if (j >= params.stateSize) return;
    
    float amp_i_r = real[i]; float amp_i_i = imag[i];
    float amp_j_r = real[j]; float amp_j_i = imag[j];
    
    // Matrix elements
    float m00_r = gate.matrix[0].x; float m00_i = gate.matrix[0].y;
    float m01_r = gate.matrix[1].x; float m01_i = gate.matrix[1].y;
    float m10_r = gate.matrix[2].x; float m10_i = gate.matrix[2].y;
    float m11_r = gate.matrix[3].x; float m11_i = gate.matrix[3].y;
    
    // new_i = m00*amp_i + m01*amp_j
    float new_i_r = (m00_r * amp_i_r - m00_i * amp_i_i) + (m01_r * amp_j_r - m01_i * amp_j_i);
    float new_i_i = (m00_r * amp_i_i + m00_i * amp_i_r) + (m01_r * amp_j_i + m01_i * amp_j_r);
    
    float new_j_r = (m10_r * amp_i_r - m10_i * amp_i_i) + (m11_r * amp_j_r - m11_i * amp_j_i);
    float new_j_i = (m10_r * amp_i_i + m10_i * amp_i_r) + (m11_r * amp_j_i + m11_i * amp_j_r);
    
    real[i] = new_i_r;
    imag[i] = new_i_i;
    real[j] = new_j_r;
    imag[j] = new_j_i;
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

/**
 * Apply General 3-Qubit Gate (CCX, CSWAP, etc.)
 */
kernel void apply_gate_3q(
    device float* real [[buffer(0)]],
    device float* imag [[buffer(1)]],
    constant GateMatrix3Q& gate [[buffer(2)]],
    constant GateParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    // Qubits: target(L), control(M), extra(H)? No, we just have 3 indices.
    // params.targetQubit, params.controlQubit, params.extraQubit
    // Sort them to insert bits in order.
    
    uint q0 = params.targetQubit; 
    uint q1 = params.controlQubit;
    uint q2 = params.extraQubit;
    
    // Sort: qA < qB < qC
    uint qa = min(q0, min(q1, q2));
    uint qc = max(q0, max(q1, q2));
    uint qb = q0 + q1 + q2 - qa - qc;
    
    // Insert 3 bits into gid. gid has N-3 bits.
    // 1. Insert 0 at qa
    uint idx = gid;
    uint maskA = (1u << qa) - 1;
    idx = ((idx & ~maskA) << 1) | (idx & maskA);
    
    // 2. Insert 0 at qb
    uint maskB = (1u << qb) - 1;
    idx = ((idx & ~maskB) << 1) | (idx & maskB);
    
    // 3. Insert 0 at qc
    uint maskC = (1u << qc) - 1;
    idx = ((idx & ~maskC) << 1) | (idx & maskC);
    
    uint base = idx;
    
    // Generate 8 indices
    // Order matters for matrix multiplication. 
    // Standard convention: |q2 q1 q0> ? 
    // Usually gate matrix rows are |000> ... |111>
    // Where "001" means LSB is 1.
    // Qiskit: q0 is LSB. 
    // So row 1 (|001>) means bit q0 is 1.
    
    uint b0 = 1u << q0;
    uint b1 = 1u << q1;
    uint b2 = 1u << q2;
    
    uint indices[8];
    // 000
    indices[0] = base;
    // 001 (bit q0 set)
    indices[1] = base | b0;
    // 010 (bit q1 set)
    indices[2] = base | b1;
    // 011
    indices[3] = base | b1 | b0;
    // 100 (bit q2 set)
    indices[4] = base | b2;
    // 101
    indices[5] = base | b2 | b0;
    // 110
    indices[6] = base | b2 | b1;
    // 111
    indices[7] = base | b2 | b1 | b0;
    
    if (indices[7] >= params.stateSize) return;
    
    // Load
    Complex amps[8];
    for(int k=0; k<8; k++) {
        amps[k] = Complex(real[indices[k]], imag[indices[k]]);
    }
    
    // Multiply
    Complex new_amps[8];
    for(int r=0; r<8; r++) {
        new_amps[r] = Complex(0,0);
        for(int c=0; c<8; c++) {
            float2 val = gate.matrix[r*8 + c];
            new_amps[r] = cadd(new_amps[r], cmul(Complex(val.x, val.y), amps[c]));
        }
    }
    
    // Store
    for(int k=0; k<8; k++) {
        real[indices[k]] = new_amps[k].real;
        imag[indices[k]] = new_amps[k].imag;
    }
}
