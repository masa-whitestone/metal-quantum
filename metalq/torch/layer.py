"""
QuantumLayer for PyTorch Neural Networks.
Integrates variational circuits as standard PyTorch layers.
"""
import torch
import torch.nn as nn
from typing import Optional, List

from ..circuit import Circuit
from ..spin import Hamiltonian
from .function import QuantumFunction
from ..backends.mps.backend import MPSBackend
from ..backends.cpu.backend import CPUBackend

class QuantumLayer(nn.Module):
    """
    A PyTorch layer representing a parameterized quantum circuit.
    
    Args:
        circuit (Circuit): The parameterized quantum circuit.
        hamiltonian (Hamiltonian): The observable to measure.
        backend_name (str): 'mps' (default) or 'cpu'.
    """
    def __init__(self, circuit: Circuit, hamiltonian: Hamiltonian, backend_name='mps'):
        super().__init__()
        self.circuit = circuit
        self.hamiltonian = hamiltonian
        
        # Initialize Backend
        if backend_name == 'mps':
            self.backend = MPSBackend()
        else:
            self.backend = CPUBackend()
            
        # Use circuit.parameters to find all unique parameters
        # This correctly handles parameters inside Expressions (recursion).
        unique_params = circuit.parameters # List[Parameter]
        
        self.param_names = [p.name for p in unique_params]
        # self.params_ref = unique_params # Keep ref if needed? No, we rely on order.
        
        init_values = []
        for _ in unique_params:
             init_values.append(torch.rand(1).item() * 6.28)

        # Create trainable parameter tensor
        self.weights = nn.Parameter(torch.tensor(init_values, dtype=torch.float32))
        
    def forward(self, x=None):
        """
        Forward pass.
        args:
            x: Input tensor (batch_size, input_dim). 
               Currently V1 supports scalar variational weights (simple optimization).
               Data re-uploading (x input) requires mixing x with weights.
               
               For V1 MVP optimization (VQE style): x is ignored or None.
               Strictly optimizing Circuit parameters.
        """
        # For simple optimization, we just use self.weights
        # QuantumFunction expects (ctx, params, circuit, H, backend)
        
        # Note: QuantumFunction.apply is how we call it.
        # Call apply() on the class, passing context automatically handles arguments.
        # Actually in Pytorch: Function.apply(input, *args)
        
        exp_val = QuantumFunction.apply(self.weights, self.circuit, self.hamiltonian, self.backend)
        
        return exp_val
