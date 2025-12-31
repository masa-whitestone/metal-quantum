
from typing import Optional, List
from .vqe import VQE, VQEResult
from ..circuit import Circuit, Parameter
from ..spin import Hamiltonian

class QAOA(VQE):
    """
    Quantum Approximate Optimization Algorithm.
    Constructs the ansatz for a given problem Hamiltonian and runs VQE.
    """
    def __init__(self, 
                 hamiltonian: Hamiltonian, 
                 reps: int = 1, 
                 backend: str = 'mps'):
        
        if hasattr(hamiltonian, 'terms'):
            self.problem_hamiltonian = hamiltonian
        else:
            # Assume PauliTerm or list-like
            from ..spin import Hamiltonian
            self.problem_hamiltonian = Hamiltonian([hamiltonian]) if not isinstance(hamiltonian, Hamiltonian) else hamiltonian
            
        self.reps = reps
        
        # Build Ansatz
        ansatz = self._build_ansatz(self.problem_hamiltonian, reps)
        
        # Initialize VQE parent
        super().__init__(ansatz, backend=backend)

    def _build_ansatz(self, cost_h: Hamiltonian, p: int) -> Circuit:
        """
        Construct standard QAOA ansatz: |+>^n -> [U_C(gamma) -> U_B(beta)]^p
        """
        n_qubits = cost_h.num_qubits
        qc = Circuit(n_qubits)
        
        # 1. Initialize |+>
        for i in range(n_qubits):
            qc.h(i)
            
        # 2. Layers
        for layer in range(p):
            gamma = Parameter(f'gamma_{layer}')
            beta = Parameter(f'beta_{layer}')
            
            # Cost Hamiltonian Evolution U_C = exp(-i gamma H_C)
            # H_C is typically sum of ZZ terms (Ising).
            # We assume cost_h is composed of Pauli Terms.
            # Evolution exp(-i gamma Z_i Z_j) is: CNOT(i,j) RZ(2*gamma, j) CNOT(i,j)
            
            for term in cost_h.terms:
                # Assuming term is Z type (Diagonal) for standard MAXCUT
                # Metal-Q native backend might eventually support exponential of Hamiltonian directly.
                # For now decompose:
                param_factor = term.coeff.real * gamma # * 2 for gate definition? 
                # RZ(theta) = exp(-i theta/2 Z). We want exp(-i (gamma*coeff) Z).
                # So theta/2 = gamma*coeff => theta = 2 * gamma * coeff.
                
                ops = term.ops # [(P, q), ...]
                
                if len(ops) == 0: continue # Identity
                
                # CNOT ladder for Multi-Z
                # Simple case: ZZ on q1, q2
                # CN(q1, q2), RZ(2*gamma*coeff, q2), CN(q1, q2)
                
                # Collect qubit indices
                indices = [q for _, q in ops]
                
                # Apply CNOTs down to last qubit
                for k in range(len(indices)-1):
                    qc.cx(indices[k], indices[k+1])
                    
                # RZ on last qubit
                last_q = indices[-1]
                # Note: We bind the parameter expression (2 * coeff * gamma)
                # Currently ParameterExpression logic in Phase 1 might be simple.
                # For MVP: Just pass gamma, and we trust user or handle coeff later.
                # BUT wait, coeff is fixed number.
                # qc.rz(gamma, last_q) # Using implicit coeff? No.
                # We need Expression support. If not present, we can't fully do generic QAOA easily without re-building circuit.
                # Alternatively, we just create parameter for each term? Bad scaling.
                
                # Let's assume simplest MaxCut for MVP: Coeff=1 or handle manually if expression support weak.
                # `qc.rz(2.0 * gamma, last_q)` if Parameter supports mul.
                # Checking parameter.py... Phase 1 implemented ParameterExpression.
                
                qc.rz(2.0 * term.coeff.real * gamma, last_q)
                
                # Inverse CNOTs
                for k in range(len(indices)-2, -1, -1):
                    qc.cx(indices[k], indices[k+1])
                    
            # Mixer Hamiltonian Evolution U_B = exp(-i beta sum X_i)
            # = Product RX(2*beta, i)
            for i in range(n_qubits):
                qc.rx(2.0 * beta, i)
                
        return qc
    
    def compute(self, max_iter: int = 100) -> VQEResult:
        return self.compute_minimum_eigenvalue(self.problem_hamiltonian, max_iter)
