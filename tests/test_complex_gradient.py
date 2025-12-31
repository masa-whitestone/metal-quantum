
import numpy as np
import metalq as mq
from metalq import Circuit, Parameter, Z, X
from metalq.backends.mps.backend import MPSBackend

def test_complex_gradient():
    print("Testing Complex Gradient (Multi-Term H, Multi-Param)...")
    
    # 1. Circuit
    # 2 ops on 2 qubits
    theta0 = Parameter('theta0')
    theta1 = Parameter('theta1') 
    theta2 = Parameter('theta2')
    
    qc = Circuit(2)
    qc.rx(theta0, 0)
    qc.ry(theta1, 1)
    qc.cx(0, 1)
    qc.rz(theta2, 1)
    
    # Parameters
    params = [0.5, 0.3, 0.8] # Randomish
    
    # Subtest 1: H = Z0
    print("\n--- Subtest 1: H = Z(0) ---")
    H1 = Z(0)
    check_gradient(qc, H1, params, "H=Z0")

    # Subtest 2: H = Z0 Z1
    print("\n--- Subtest 2: H = Z(0) Z(1) ---")
    H2 = Z(0) * Z(1)
    check_gradient(qc, H2, params, "H=Z0Z1")
    
    # Subtest 3: H = Z0 Z1 + 0.5 X0
    print("\n--- Subtest 3: H = Z0Z1 + 0.5X0 ---")
    H3 = (Z(0) * Z(1)) + (0.5 * X(0))
    check_gradient(qc, H3, params, "H=Full")
    
def check_gradient(qc, H, params, name):
    bk_mps = MPSBackend()
    try:
        grads_adj = bk_mps.gradient(qc, H, params, method='adjoint')
        # print(f"Adjoint Gradients ({name}):", grads_adj)
    except Exception as e:
        print(f"Adjoint ({name}) failed:", e)
        return

    from metalq.backends.cpu.backend import CPUBackend
    bk_cpu = CPUBackend()
    grads_ps = bk_cpu.gradient(qc, H, params, method='parameter_shift')
    # print(f"ParamShift Gradients ({name}):", grads_ps)
    
    diff = np.linalg.norm(grads_adj - grads_ps)
    print(f"Difference ({name}): {diff}")
    
    if diff < 1e-4:
        print(f"SUCCESS ({name})")
    else:
        print(f"FAILURE ({name})")
        print(f"Adjoint: {grads_adj}")
        print(f"Ref:     {grads_ps}")

if __name__ == "__main__":
    test_complex_gradient()
