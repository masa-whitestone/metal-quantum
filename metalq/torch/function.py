"""
QuantumFunction for PyTorch Autograd Integration.
Connects Metal-Q backends (especially MPS Adjoint Diff) to PyTorch's backward engine.
"""
import torch
import numpy as np
from typing import List, Optional

from ..parameter import Parameter
from ..circuit import Circuit
from ..spin import Hamiltonian

class QuantumFunction(torch.autograd.Function):
    """
    Custom autograd function for variational quantum circuits.
    Forward: Calculates Expectation Value <H>.
    Backward: Calculates Gradients via Adjoint Differentiation (or Parameter Shift).
    """
    
    @staticmethod
    def forward(ctx, params_tensor, circuit, hamiltonian, backend):
        """
        Args:
            ctx: Context object to save info for backward.
            params_tensor: Tensor of shape (n_params,) containing parameter values.
            circuit: Metal-Q Circuit object (parameterized).
            hamiltonian: Metal-Q Hamiltonian/PauliTerm.
            backend: Initialized Metal-Q backend instance.
        """
        # Save context for backward
        ctx.circuit = circuit
        ctx.hamiltonian = hamiltonian
        ctx.backend = backend
        
        # Convert tensor to list for backend
        params_list = params_tensor.detach().numpy().tolist()
        ctx.params_list = params_list
        
        # Calculate Expectation
        exp_val = backend.expectation(circuit, hamiltonian, params_list)
        
        # Return as tensor
        ctx.save_for_backward(params_tensor)
        return torch.tensor(exp_val, dtype=params_tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using Adjoint Differentiation.
        """
        params_tensor, = ctx.saved_tensors
        circuit = ctx.circuit
        hamiltonian = ctx.hamiltonian
        backend = ctx.backend
        
        # We need gradients w.r.t parameters
        # Call backend.gradient
        # If backend supports 'adjoint', it will be fast. 
        # Only MPS supports 'adjoint' currently.
        
        method = 'adjoint' if backend.name == 'mps' else 'parameter_shift'
        
        grads = backend.gradient(circuit, hamiltonian, ctx.params_list, method=method)
        
        # Chain Rule Mapping (Gate Grads -> Circuit Param Grads)
        # backend.gradient returns flat array of gradients for every gate parameter.
        
        unique_params = circuit.parameters
        num_inputs = len(unique_params)
        final_grads = np.zeros(num_inputs)
        
        gate_ptr = 0
        from ..parameter import Parameter, ParameterExpression
        
        for gate in circuit.gates:
            for p in gate.params:
                # Backend gradient for this gate parameter
                g_gate = grads[gate_ptr]
                gate_ptr += 1
                
                # Distribution to input parameters (Chain Rule)
                # dL/dInput = dL/dGate * dGate/dInput
                
                if isinstance(p, Parameter):
                    # Direct parameter
                    # Find index in unique_params (inefficient search? Map would be better)
                    # Optimization: Create map once in Forward? 
                    # For MVP, linear scan or check.
                    try:
                        idx = unique_params.index(p)
                        final_grads[idx] += g_gate
                    except ValueError:
                        pass # Parameter not in inputs?
                        
                elif isinstance(p, ParameterExpression):
                    # Expression: dGate/dInput comes from p.grad(up)
                    for i, up in enumerate(unique_params):
                        scale = p.grad(up)
                        if abs(scale) > 1e-9:
                            final_grads[i] += g_gate * scale
                            
        # Convert to tensor
        grads_tensor = torch.from_numpy(final_grads).to(params_tensor.dtype)
        
        # Chain rule: dL/dParam = dL/dExp * dExp/dParam
        final_grad = grad_output * grads_tensor
        
        # Return gradient for each input to forward. None for non-tensors.
        return final_grad, None, None, None
