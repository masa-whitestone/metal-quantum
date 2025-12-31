
from typing import Optional, Union, List, Dict
import numpy as np
import torch
import torch.optim as optim

from ..circuit import Circuit
from ..spin import Hamiltonian
from ..torch.layer import QuantumLayer

class VQEResult:
    def __init__(self, eigenvalue: float, optimal_params: List[float], history: List[float]):
        self.eigenvalue = eigenvalue
        self.optimal_params = optimal_params
        self.history = history

class VQE:
    """
    Variational Quantum Eigensolver implementation using Metal-Q Adjoint Differentiation.
    
    Args:
        ansatz (Circuit): Parameterized quantum circuit.
        optimizer_cls (torch.optim.Optimizer): Optimizer class (default: Adam).
        optimizer_kwargs (dict): Arguments for optimizer (default: lr=0.1).
        backend (str): Backend name ('mps' or 'cpu').
    """
    def __init__(self, 
                 ansatz: Circuit, 
                 optimizer_cls=optim.Adam, 
                 optimizer_kwargs: Optional[Dict] = None,
                 backend: str = 'mps'):
        self.ansatz = ansatz
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {'lr': 0.1}
        self.backend_name = backend

    def compute_minimum_eigenvalue(self, hamiltonian: Hamiltonian, max_iter: int = 100, tol: float = 1e-4) -> VQEResult:
        """
        Run the VQE optimization loop.
        
        Args:
            hamiltonian: The Hamiltonian to minimize.
            max_iter: Maximum optimization steps.
            tol: Convergence tolerance.
            
        Returns:
            VQEResult object.
        """
        # 1. Setup Quantum Layer
        layer = QuantumLayer(self.ansatz, hamiltonian, backend_name=self.backend_name)
        
        # 2. Setup Optimizer
        optimizer = self.optimizer_cls(layer.parameters(), **self.optimizer_kwargs)
        
        # 3. Optimization Loop
        history = []
        best_loss = float('inf')
        patience = 5
        no_improv = 0
        
        for i in range(max_iter):
            optimizer.zero_grad()
            loss = layer()
            loss.backward()
            optimizer.step()
            
            val = loss.item()
            history.append(val)
            
            # Simple early stopping
            if abs(val - best_loss) < tol:
                no_improv += 1
            else:
                no_improv = 0
                
            if val < best_loss:
                best_loss = val
                
            if no_improv >= patience:
                break
                
        # 4. Extract results
        opt_params = layer.weights.detach().numpy().tolist()
        
        return VQEResult(best_loss, opt_params, history)
