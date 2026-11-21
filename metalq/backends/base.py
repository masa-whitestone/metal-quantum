"""
metalq/backends/base.py - Abstract Backend Base Class

すべてのバックエンドが実装すべきインターフェースを定義。
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..circuit import Circuit
    from ..spin import Hamiltonian


def gate_slot_grads_to_parameter_grads(circuit: 'Circuit',
                                       slot_grads: Sequence[float]
                                       ) -> np.ndarray:
    """Chain-rule per-gate-parameter-slot gradients onto circuit parameters.

    Backends whose kernels differentiate every *gate parameter slot*
    (one entry per entry of ``gate.params``, in gate order, constants
    included) use this to produce the array the Backend gradient
    contract asks for: one entry per unique circuit parameter, in
    ``circuit.parameters`` order.

    A slot holding a bare ``Parameter`` contributes its gradient to that
    parameter; a slot holding a ``ParameterExpression`` contributes
    ``expr.grad(p)`` times the slot gradient to each parameter ``p`` it
    involves (so ``rz(2*gamma)`` contributes 2x, and a parameter reused
    by several gates accumulates every gate's share); a slot holding a
    plain number is consumed and ignored.

    Args:
        circuit: The *unbound* parameterized circuit the gradients were
            computed for.
        slot_grads: dE/d(gate parameter slot), length
            ``sum(len(g.params) for g in circuit.gates)``.

    Returns:
        np.ndarray of shape (len(circuit.parameters),).

    Raises:
        ValueError: if ``slot_grads`` does not have one entry per gate
            parameter slot.
        NotImplementedError: if a slot holds a non-linear expression
            (``a * b``), which has no parameter-independent chain-rule
            coefficient. Adjoint differentiation refuses these circuits
            for the same reason, so neither path can differentiate them;
            failing loudly beats returning a wrong gradient.
    """
    from ..parameter import Parameter, ParameterExpression

    gates = circuit.gates
    expected = sum(len(g.params) for g in gates)
    flat = np.asarray(slot_grads, dtype=float).ravel()
    if flat.size != expected:
        raise ValueError(
            f"expected {expected} gate-parameter-slot gradients "
            f"(sum of len(gate.params)), got {flat.size}")

    plist = circuit.parameters
    index = {p: i for i, p in enumerate(plist)}
    out = np.zeros(len(plist))
    ptr = 0
    for gate in gates:
        for p in gate.params:
            g = flat[ptr]
            ptr += 1
            if isinstance(p, Parameter):
                idx = index.get(p)
                if idx is not None:
                    out[idx] += g
            elif isinstance(p, ParameterExpression):
                for q in p.parameters:
                    idx = index.get(q)
                    if idx is None:
                        continue
                    try:
                        scale = p.grad(q)
                    except NotImplementedError as exc:
                        raise NotImplementedError(
                            f"cannot differentiate gate parameter '{p}' "
                            f"with respect to '{q}': {exc}") from exc
                    if scale:
                        out[idx] += g * scale
            # plain float slot: consumed, contributes to no parameter
    return out


class Backend(ABC):
    """
    Abstract base class for quantum simulation backends.
    
    すべてのバックエンド (CPU, MPS) はこのクラスを継承し、
    以下のメソッドを実装する必要がある。
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g., 'cpu', 'mps')."""
        pass
    
    @property
    @abstractmethod
    def max_qubits(self) -> int:
        """Maximum supported qubits (memory-limited)."""
        pass
    
    @abstractmethod
    def run(self, 
            circuit: 'Circuit', 
            shots: int = 0,
            params: Optional[Union[Dict, List[float]]] = None) -> Dict:
        """
        Execute a quantum circuit.
        
        Args:
            circuit: Circuit to execute
            shots: Number of measurement shots.
                   0 = return statevector only
                   >0 = sample and return counts
            params: Parameter values for parameterized circuits
        
        Returns:
            Dict containing:
              - 'statevector': np.ndarray (if shots=0)
              - 'counts': Dict[str, int] (if shots>0)
              - 'time_ms': float (execution time)
        """
        pass
    
    @abstractmethod
    def statevector(self, 
                    circuit: 'Circuit',
                    params: Optional[Union[Dict, List[float]]] = None) -> np.ndarray:
        """
        Get the final statevector after circuit execution.
        
        Args:
            circuit: Circuit to execute (without measurements)
            params: Parameter values
        
        Returns:
            Complex statevector of shape (2^n,)
        """
        pass
    
    @abstractmethod
    def expectation(self, 
                    circuit: 'Circuit', 
                    hamiltonian: 'Hamiltonian',
                    params: Optional[Union[Dict, List[float]]] = None) -> float:
        """
        Compute expectation value <ψ|H|ψ>.
        
        Args:
            circuit: Circuit preparing state |ψ⟩
            hamiltonian: Observable H
            params: Parameter values
        
        Returns:
            Real expectation value
        """
        pass
    
    @abstractmethod
    def gradient(self, 
                 circuit: 'Circuit', 
                 hamiltonian: 'Hamiltonian',
                 params: List[float],
                 method: str = 'parameter_shift') -> np.ndarray:
        """
        Compute gradient of expectation value w.r.t. parameters.
        
        Args:
            circuit: Parameterized circuit
            hamiltonian: Observable
            params: Current parameter values
            method: Differentiation method
                    - 'parameter_shift': 2回路/パラメータ
                    - 'adjoint': GPU向け高速アルゴリズム
        
        Returns:
            Gradient array of shape ``(len(circuit.parameters),)``.

        Contract (all backends, all methods):
            One entry per **unique circuit parameter**, ordered by
            ``circuit.parameters`` (first appearance) -- the same order
            ``params`` and ``circuit.bind_parameters(params)`` use, so
            ``gradient()[i]`` is dE/d(params[i]).

            This is NOT the per-gate-parameter-slot layout some kernels
            produce natively: a parameter driving several gates gets the
            summed contribution, an expression slot like ``rz(2*gamma)``
            is chain-ruled through ``ParameterExpression.grad``, and
            constant-valued gate parameters contribute nothing. Backends
            whose kernels return per-slot values must map them with
            ``gate_slot_grads_to_parameter_grads`` before returning.
        """
        pass

    def expectation_and_gradient(self,
                                  circuit: 'Circuit',
                                  hamiltonian: 'Hamiltonian',
                                  params: List[float]):
        """
        Compute the expectation value and its gradient together.

        Default implementation: calls expectation() then gradient() (two
        forward passes). Backends that can fuse both into a single GPU pass
        (e.g. MPSBackend) should override this for a real speedup.

        Args:
            circuit: Parameterized circuit
            hamiltonian: Observable
            params: Current parameter values

        Returns:
            (energy, gradient) tuple: energy is a float, gradient is an
            np.ndarray of shape ``(len(circuit.parameters),)`` following
            the same contract as ``gradient()``.
        """
        energy = self.expectation(circuit, hamiltonian, params)
        grad = self.gradient(circuit, hamiltonian, params)
        return energy, grad

    def sample(self,
               circuit: 'Circuit', 
               shots: int,
               params: Optional[Union[Dict, List[float]]] = None) -> Dict[str, int]:
        """
        Sample measurement results.
        
        Convenience method that calls run() with shots > 0.
        """
        result = self.run(circuit, shots=shots, params=params)
        return result.get('counts', {})
