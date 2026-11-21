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
            ctx (torch.autograd.function.FunctionCtx): Context object to save info for backward.
            params_tensor (torch.Tensor): Tensor of shape (n_params,) containing parameter values.
            circuit (Circuit): Metal-Q Circuit object (parameterized).
            hamiltonian (Hamiltonian): Metal-Q Hamiltonian/PauliTerm.
            backend (Backend): Initialized Metal-Q backend instance.
        """
        # Save context for backward
        ctx.circuit = circuit
        ctx.hamiltonian = hamiltonian
        ctx.backend = backend

        # Convert tensor to list for backend
        params_list = params_tensor.detach().numpy().tolist()
        ctx.params_list = params_list

        # If a gradient will be needed, fuse the forward pass that computes it
        # with the energy evaluation (avoids running the circuit twice: once
        # here, once again in backward() via backend.gradient()). Backends
        # that can't fuse (or don't need to) fall back internally to plain
        # expectation() + gradient(), so this is always safe to call.
        if ctx.needs_input_grad[0]:
            exp_val, grads = backend.expectation_and_gradient(circuit, hamiltonian, params_list)
            ctx.grads = grads
        else:
            exp_val = backend.expectation(circuit, hamiltonian, params_list)
            ctx.grads = None

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

        if ctx.grads is not None:
            # Already computed in forward() via the fused expectation_and_gradient
            # path -- no need to run the circuit a second time here.
            grads = ctx.grads
        else:
            # forward() didn't anticipate needing a gradient (e.g. called under
            # a no-grad context that later got overridden); compute it now.
            # Every backend accepts method='adjoint' and falls back internally
            # (CPU: to parameter-shift; MPS: to parameter-shift for gates its
            # native kernel can't differentiate).
            grads = backend.gradient(circuit, hamiltonian, ctx.params_list,
                                     method='adjoint')

        # backend.gradient() already returns one entry per unique circuit
        # parameter, in circuit.parameters order == the order of the input
        # tensor (see Backend.gradient's contract). Backends whose kernels
        # produce per-gate-parameter-slot values map them through
        # backends.base.gate_slot_grads_to_parameter_grads themselves, so no
        # chain-rule pass is needed here.
        final_grads = np.asarray(grads, dtype=float).ravel()
        if final_grads.size != len(circuit.parameters):
            raise ValueError(
                f"backend '{backend.name}' returned {final_grads.size} "
                f"gradients for {len(circuit.parameters)} circuit parameters")

        # Convert to tensor
        grads_tensor = torch.from_numpy(final_grads).to(params_tensor.dtype)
        
        # Chain rule: dL/dParam = dL/dExp * dExp/dParam
        final_grad = grad_output * grads_tensor
        
        # Return gradient for each input to forward. None for non-tensors.
        return final_grad, None, None, None
