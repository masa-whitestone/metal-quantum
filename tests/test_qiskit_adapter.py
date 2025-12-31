
import numpy as np
import metalq as mq
from metalq.adapters.qiskit_adapter import to_metalq

try:
    from qiskit import QuantumCircuit
except ImportError:
    print("Skipping Qiskit test (not installed)")
    exit(0)

def test_qiskit_conversion():
    print("Testing Qiskit -> MetalQ Conversion...")
    
    # 1. Create Qiskit Circuit (Bell State)
    qc_qiskit = QuantumCircuit(2)
    qc_qiskit.h(0)
    qc_qiskit.cx(0, 1)
    # Add some rotations to test param mapping
    qc_qiskit.rx(0.5, 0)
    qc_qiskit.rx(-0.5, 0) # Undo
    
    # 2. Convert
    qc_metalq = to_metalq(qc_qiskit)
    
    # 3. Verify Structure
    print("Metal-Q Circuit Gate Count:", len(qc_metalq.gates))
    
    # 4. Run on MetalQ
    # Expectation of ZZ should be 1.0 (Bell State)
    H = mq.Z(0) * mq.Z(1)
    val = mq.expect(qc_metalq, H, backend='mps')
    
    print(f"Expectation <ZZ>: {val}")
    
    if abs(val - 1.0) < 1e-4:
        print("SUCCESS: Adapter conversion verified.")
    else:
        print("FAILURE: Incorrect result.")

if __name__ == "__main__":
    test_qiskit_conversion()
