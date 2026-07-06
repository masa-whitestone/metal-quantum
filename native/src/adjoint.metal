/*
 * adjoint.metal - Reduction and Fused Pauli-String Kernels
 *
 * Fused Pauli-string kernels exploit <i|P|j> being nonzero only for
 * j = i ^ x_mask, with matrix element g * (-1)^popcount(i & (y|z)) where
 * g = (-i)^nY is a per-term constant (folded in on the host). This lets
 * every Pauli-term operation run in ONE pass over the statevector — no
 * scratch copy, no per-qubit gate dispatches.
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

// ===========================================================================
// Fused Pauli-string kernels
// ===========================================================================
// Tree reductions use 256 threads with a barrier at EVERY step: the
// barrier-less "warp-synchronous" tail is not valid on Apple GPUs (the
// compiler may cache threadgroup memory in registers).

// Per-term parameters for a Pauli string P = prod_q P_q.
struct PauliTermParams {
    uint x_mask;   // qubits with X or Y (bit flip)
    uint y_mask;   // qubits with Y
    uint z_mask;   // qubits with Z
    uint total_elements;
    float g_re;    // g = (-i)^popcount(y_mask), premultiplied by any host coeff
    float g_im;
};

// Sign (-1)^popcount(i & (y|z)) as a float.
inline float pauli_sign(uint id, uint y_mask, uint z_mask) {
    return (popcount(id & (y_mask | z_mask)) & 1u) ? -1.0f : 1.0f;
}

// partial_sums[gid] = sum over group of Re[ g * s_i * conj(psi_i) * psi_{i^x} ]
// One pass computes <psi|P|psi> partials for ANY Pauli string; the Z-string
// case (x_mask == 0) reduces to a signed-probability sum.
kernel void pauli_expectation(
    device const Complex* psi [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
    constant PauliTermParams& p [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_mem[256];

    float term = 0.0f;
    if (id < p.total_elements) {
        Complex a = psi[id];
        Complex b = psi[id ^ p.x_mask];
        Complex c = z_mul(z_conj(a), b);
        term = pauli_sign(id, p.y_mask, p.z_mask) *
               (p.g_re * c.real - p.g_im * c.imag);
    }

    shared_mem[tid] = term;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial_sums[gid] = shared_mem[0];
    }
}

// partial_sums[gid] = sum over group of conj(bra_i) * (g * s_i * ket_{i^x})
// Complex-valued variant for gradient overlaps <lambda|P|psi>.
kernel void pauli_inner_product(
    device const Complex* bra [[buffer(0)]],
    device const Complex* ket [[buffer(1)]],
    device Complex* partial_sums [[buffer(2)]],
    constant PauliTermParams& p [[buffer(3)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    threadgroup Complex shared_mem[256];

    Complex term = {0.0f, 0.0f};
    if (id < p.total_elements) {
        Complex l = bra[id];
        Complex k = ket[id ^ p.x_mask];
        Complex g = {p.g_re, p.g_im};
        float s = pauli_sign(id, p.y_mask, p.z_mask);
        Complex pk = z_mul(g, k);
        pk.real *= s;
        pk.imag *= s;
        term = z_mul(z_conj(l), pk);
    }

    shared_mem[tid] = term;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] = z_add(shared_mem[tid], shared_mem[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial_sums[gid] = shared_mem[0];
    }
}

// ===========================================================================
// Measurement sampling: per-block probability reduction
// ===========================================================================
// block_sums[gid] = sum over the 256-element block of |psi_i|^2 = re^2 + im^2.
// Same 256-wide tree-reduction structure as pauli_expectation (barrier at
// EVERY step, uniform threadgroups, out-of-range threads contribute 0). The
// host builds a coarse CDF over these block sums, so a shot is resolved by a
// binary search over blocks + a walk within one 256-element block — the full
// statevector never leaves the GPU.
kernel void block_probability_sums(
    device const Complex* psi [[buffer(0)]],
    device float* block_sums [[buffer(1)]],
    constant uint& total_elements [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_mem[256];

    float prob = 0.0f;
    if (id < total_elements) {
        Complex a = psi[id];
        prob = a.real * a.real + a.imag * a.imag;
    }

    shared_mem[tid] = prob;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        block_sums[gid] = shared_mem[0];
    }
}

// dst[i] += g * s_i * src[i ^ x_mask]   (g carries the term coefficient)
// One pass replaces copy + per-qubit Pauli gates + linear combine when
// accumulating H|psi> term by term.
kernel void pauli_accumulate(
    device const Complex* src [[buffer(0)]],
    device Complex* dst [[buffer(1)]],
    constant PauliTermParams& p [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    Complex k = src[id ^ p.x_mask];
    Complex g = {p.g_re, p.g_im};
    float s = pauli_sign(id, p.y_mask, p.z_mask);
    Complex prod = z_mul(g, k);
    prod.real *= s;
    prod.imag *= s;
    dst[id] = z_add(dst[id], prod);
}
