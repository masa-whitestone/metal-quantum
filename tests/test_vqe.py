
import metalq as mq
from metalq.algorithms.vqe import VQE
from metalq import Circuit, Parameter, Z

def test_vqe():
    print("Testing VQE Algorithm...")
    
    # 1. Simple Ansatz: RY(theta)
    theta = Parameter('theta')
    qc = mq.Circuit(1)
    qc.ry(theta, 0)
    
    # 2. Hamiltonian: Z (Min eigenval -1 at theta=pi)
    H = Z(0)
    
    # 3. VQE
    vqe = VQE(qc, backend='mps')
    
    # 4. Run
    # RY(0) = |0>, <Z>=1. RY(pi)=|1>, <Z>=-1.
    result = vqe.compute_minimum_eigenvalue(H, max_iter=50)
    
    print(f"Min Eigenvalue: {result.eigenvalue}")
    print(f"Opt Params: {result.optimal_params}")
    
    if result.eigenvalue < -0.99:
        print("SUCCESS: VQE found ground state.")
    else:
        print("FAILURE: VQE did not converge.")

if __name__ == "__main__":
    test_vqe()
