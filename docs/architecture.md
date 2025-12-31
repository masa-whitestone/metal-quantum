# Architecture

Metal-Q is built with a layered architecture to maximize performance while maintaining ease of use:

1.  **Python API**: High-level interface (`metalq`) and PyTorch bindings.
2.  **C Interface**: Lightweight Ctypes bridge (`native/include/metalq.h`).
3.  **Objective-C Native Layer**: Manages Metal context (`MTLDevice`, `MTLCommandQueue`) and buffers.
4.  **Metal Compute Shaders**: Optimized GPU kernels written in Metal Shading Language (MSL).

## Components

```mermaid
graph TD
    A[User Code] --> B[metalq (Python)]
    B --> C[metalq.backend.mps]
    C --> D[libmetalq.dylib (C/Obj-C)]
    D --> E[Metal API]
    E --> F[GPU Compute Shaders]
    
    B --> G[PyTorch Integration]
    G -.-> H[Autograd]
    G --> B
```

## Gate Execution Flow

1.  User defines `Circuit`.
2.  `run(circuit)` is called.
3.  Backend converts gate list to C structs (`mq_gate_t`).
4.  `metalq_run` is invoked via ctypes.
5.  Native layer uploads state vector and gate parameters to GPU buffers.
6.  Compute shaders apply gates in parallel (one thread per amplitude pair).
7.  Results are downloaded back to CPU (if needed).
