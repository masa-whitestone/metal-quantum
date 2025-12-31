
import ctypes
import numpy as np
import os
from metalq.backends.mps.backend import MPSBackend, MQComplex, MQGate
from metalq import Circuit

# Initialize backend
backend = MPSBackend()
print("Backend initialized")

# Create a simple circuit
qc = Circuit(1).x(0)
print("Circuit created")

# Run native
sv = backend.statevector(qc)
print(f"Statevector: {sv}")
print(f"Abs(sv[0]): {np.abs(sv[0])}")

if np.abs(sv[0].real - 999.0) < 0.1:
    print("SUCCESS: Shader execution confirmed (force write detected)")
else:
    print("FAILURE: Shader did not write 999.0")
