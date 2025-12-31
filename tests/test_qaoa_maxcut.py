
import metalq as mq
from metalq.algorithms.qaoa import QAOA
from metalq import Z, I

def test_qaoa():
    print("Testing QAOA Algorithm (MaxCut 2-node)...")
    
    # 1. MaxCut Hamiltonian for edge (0,1)
    # H_c = 0.5 * (I - Z_0 Z_1)
    # We want to MINIMIZE -H_c (Maximize Cut) OR Minimize original Ising Model.
    # Standard Ising for MaxCut: Minimize Sum Z_i Z_j (Antiferromagnetic). 
    # If Z_i != Z_j (Cut), Z_i Z_j = -1. Minimized.
    # So we minimize H = Z_0 * Z_1.
    
    H = Z(0) * Z(1)
    
    # 2. QAOA (p=1)
    qaoa = QAOA(H, reps=1, backend='mps')
    
    # 3. Run
    # Minimum eigenval of ZZ is -1 (states |01> or |10>)
    result = qaoa.compute(max_iter=50)
    
    print(f"Min Eigenvalue: {result.eigenvalue}")
    print(f"Opt Params: {result.optimal_params}")
    
    # Expect -1.0
    if result.eigenvalue < -0.99:
        print("SUCCESS: QAOA found MaxCut solution.")
    else:
        print("FAILURE: QAOA did not converge.")

if __name__ == "__main__":
    test_qaoa()
