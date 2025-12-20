/**
 * measurement.metal - GPU-accelerated Sampling Shaders
 * 
 * Implements parallel probability computation, prefix sum, and sampling.
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================
// Kernel 1: Compute Probabilities
// prob[i] = |amp[i]|² = real[i]² + imag[i]²
// ============================================================
kernel void compute_probabilities(
    device const float* real [[buffer(0)]],
    device const float* imag [[buffer(1)]],
    device float* prob [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    float r = real[gid];
    float i = imag[gid];
    prob[gid] = r * r + i * i;
}

// ============================================================
// Kernel 2: Prefix Sum (Blelloch Scan) - Up-sweep Phase
// ============================================================
kernel void prefix_sum_up_sweep(
    device float* data [[buffer(0)]],
    constant uint& stride [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = (gid + 1) * stride * 2 - 1;
    uint prev_idx = idx - stride;
    
    if (idx < size) {
        data[idx] += data[prev_idx];
    }
}

// ============================================================
// Kernel 3: Prefix Sum - Down-sweep Phase
// ============================================================
kernel void prefix_sum_down_sweep(
    device float* data [[buffer(0)]],
    constant uint& stride [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = (gid + 1) * stride * 2 - 1;
    uint prev_idx = idx - stride;
    
    if (idx < size) {
        float temp = data[prev_idx];
        data[prev_idx] = data[idx];
        data[idx] += temp;
    }
}

// ============================================================
// GPU Random Number Generator (Xorshift128+)
// ============================================================
struct RNGState {
    ulong s0;
    ulong s1;
};

inline float gpu_random(thread RNGState& state) {
    ulong s1 = state.s0;
    ulong s0 = state.s1;
    state.s0 = s0;
    s1 ^= s1 << 23;
    state.s1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    
    ulong result = state.s1 + s0;
    return (float)(result & 0xFFFFFFFF) / (float)0x100000000;
}

inline RNGState init_rng(uint seed, uint gid) {
    RNGState state;
    state.s0 = (ulong)seed * 6364136223846793005ULL + (ulong)gid * 1442695040888963407ULL;
    state.s1 = (ulong)(seed ^ 0xDEADBEEF) * 6364136223846793005ULL + (ulong)(gid + 1) * 1442695040888963407ULL;
    
    // Warm up RNG
    for (int i = 0; i < 8; i++) {
        gpu_random(state);
    }
    return state;
}

// ============================================================
// Kernel 4: Sample and Count (Fused Kernel)
// Each thread samples multiple shots and updates histogram atomically
// ============================================================
kernel void sample_and_count(
    device const float* cumProb [[buffer(0)]],
    device atomic_uint* histogram [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    constant uint& seed [[buffer(3)]],
    constant uint& shotsPerThread [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    RNGState rng = init_rng(seed, gid);
    
    for (uint s = 0; s < shotsPerThread; s++) {
        float r = gpu_random(rng);
        
        // Binary search for inverse CDF sampling
        uint low = 0;
        uint high = size - 1;
        
        while (low < high) {
            uint mid = (low + high) / 2;
            if (cumProb[mid] < r) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        
        atomic_fetch_add_explicit(&histogram[low], 1, memory_order_relaxed);
    }
}

// ============================================================
// Kernel 5: Normalize Cumulative Probabilities
// ============================================================
kernel void normalize_cum_prob(
    device float* cumProb [[buffer(0)]],
    constant uint& size [[buffer(1)]],
    constant float& total [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    if (total > 0) {
        cumProb[gid] /= total;
    }
}

// ============================================================
// Kernel 6: Zero Histogram
// ============================================================
kernel void zero_histogram(
    device atomic_uint* histogram [[buffer(0)]],
    constant uint& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    atomic_store_explicit(&histogram[gid], 0, memory_order_relaxed);
}
