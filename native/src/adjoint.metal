/*
 * adjoint.metal - Adjoint Differentiation Kernels
 */

#include <metal_stdlib>
using namespace metal;

// Must match definitions in quantum_gates.metal and C API (FP32)
struct Complex {
    float real;
    float imag;
};

Complex z_mul(Complex a, Complex b) {
    return {a.real * b.real - a.imag * b.imag,
            a.real * b.imag + a.imag * b.real};
}

Complex z_conj(Complex a) {
    return {a.real, -a.imag};
}

Complex z_add(Complex a, Complex b) {
    return {a.real + b.real, a.imag + b.imag};
}

// Atomic float support is limited in Metal (requires Tier 2).
// We will output an array of partial sums and reduce on CPU for simplicity in v1.0.

kernel void compute_overlap(
    device const Complex* lambda [[buffer(0)]],
    device const Complex* psi [[buffer(1)]],
    device Complex* output [[buffer(2)]],
    constant uint& generator_type [[buffer(3)]], // 0=X, 1=Y, 2=Z
    uint id [[thread_position_in_grid]],
    uint total_threads [[threads_per_grid]]
) {
    if (id >= total_threads) return;
    
    Complex l = lambda[id];
    Complex p = psi[id];
    
    // Apply generator P to psi: |p'> = P |psi>
    // P is Pauli X, Y, or Z.
    // However, P acts on a specific qubit?
    // Wait, the gradient formula is <lambda | P_k | psi>.
    // P_k acts on qubit k.
    // This kernel needs to know which qubit and handle the logic (bit flip/phase).
    // This is basically "Apply Gate" logic + "Dot Product".
    
    // Re-thinking:
    // It's easier to apply P to psi using the existing "Apply Gate" kernel (conceptually),
    // storing it in a temp buffer, then do a simple dot product.
    // BUT, that requires extra memory.
    
    // Fused kernel:
    // Inputs: lambda, psi, target_qubit, generator_type
    // This kernel iterates over N/2 pairs (like apply_gate).
    // For each pair indices (idx0, idx1), it computes contribution.
    
    // ...
}

// Simpler approach for v1.0:
// Just implement a "Complex Dot Product" kernel.
// The caller (host) is responsible for applying P to psi (using apply_gate or temporarily modifying the gate matrix).
// If we modify the backward pass to apply P * U instead of U, we get P|psi>.
// Then we just need <lambda | (P|psi>)>.
// So we just need a dot product kernel.

kernel void complex_inner_product(
    device const Complex* bra [[buffer(0)]], 
    device const Complex* ket [[buffer(1)]], 
    device Complex* partial_sums [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]]
) {
    threadgroup Complex shared_mem[256];

    Complex term = {0.0f, 0.0f};

    if (id < total_elements) {
        Complex l = bra[id]; 
        Complex k = ket[id];
        term = z_mul(z_conj(l), k);
    }
    // Else term is 0.0 (padding)

    if (tid < 256) {
        shared_mem[tid] = term;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Explicit Unrolled Tree Reduction for 256 threads
    // We assume block_size is fixed to 256 by dispatch logic (uniform grid).
    
    if (tid < 128) shared_mem[tid] = z_add(shared_mem[tid], shared_mem[tid + 128]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 64) shared_mem[tid] = z_add(shared_mem[tid], shared_mem[tid + 64]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid < 32) {
        shared_mem[tid] = z_add(shared_mem[tid], shared_mem[tid + 32]);
        simdgroup_barrier(mem_flags::mem_threadgroup); // Safe measure
        shared_mem[tid] = z_add(shared_mem[tid], shared_mem[tid + 16]);
        shared_mem[tid] = z_add(shared_mem[tid], shared_mem[tid + 8]);
        shared_mem[tid] = z_add(shared_mem[tid], shared_mem[tid + 4]);
        shared_mem[tid] = z_add(shared_mem[tid], shared_mem[tid + 2]);
        shared_mem[tid] = z_add(shared_mem[tid], shared_mem[tid + 1]);
    }
    
    if (tid == 0) {
        partial_sums[gid] = shared_mem[0];
    }
}

kernel void complex_linear_combine(
    device const Complex* src [[buffer(0)]],
    device Complex* dst [[buffer(1)]],
    constant float& coeff_real [[buffer(2)]],
    constant float& coeff_imag [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    // dst[i] += coeff * src[i]
    Complex s = src[id];
    Complex d = dst[id];
    
    Complex c = {coeff_real, coeff_imag};
    Complex prod = z_mul(c, s);
    
    dst[id] = z_add(d, prod);
}
