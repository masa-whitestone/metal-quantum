"""
metalq/backends/cpu/gradient.py - Parameter Shift Gradient

パラメータシフト則を用いた勾配計算 (adjoint が使えない回路の
フォールバック; どのバックエンドからでも使える)。

シフトするのは回路パラメータ θ ではなく**ゲートのパラメータスロット**
であることに注意。d<H>/dθ = Σ_slot (d(slot)/dθ) * d<H>/d(slot) と
連鎖律で戻す。θ を直接 π/2 動かすと、rz(2γ) のような式ではゲート角が
π 動いてしまい (シフト則が成り立たず勾配が 0 になる)、1 つの θ が
複数ゲートに現れる場合も誤った値になる。
"""
import copy

import numpy as np
from typing import Any, List, TYPE_CHECKING

from ...parameter import is_parameterized
from ..base import gate_slot_grads_to_parameter_grads

if TYPE_CHECKING:
    from ...circuit import Circuit
    from ...spin import Hamiltonian
    from .backend import CPUBackend


def _gate_list(circuit) -> List[Any]:
    return circuit.gates if hasattr(circuit, 'gates') else circuit._gates


def _shifted_circuit(bound, gate_idx: int, slot: int, delta: float):
    """Copy ``bound`` with one gate parameter slot shifted by ``delta``."""
    new = copy.copy(bound)
    gates = list(bound._gates)
    gate = gates[gate_idx]
    params = list(gate.params)
    params[slot] = float(params[slot]) + delta
    gates[gate_idx] = type(gate)(name=gate.name,
                                 qubits=list(gate.qubits),
                                 params=params)
    new._gates = gates
    return new


def parameter_shift_gradient(backend: 'CPUBackend',
                             circuit: 'Circuit',
                             hamiltonian: 'Hamiltonian',
                             params: List[float],
                             shift: float = np.pi / 2) -> np.ndarray:
    """Gradient of <H> via the parameter-shift rule.

    ゲートパラメータスロット v ごとに
        d<H>/dv = ( <H>(v + s) - <H>(v - s) ) / (2 sin s)
    を評価し (s = π/2 なら 1/2)、連鎖律で回路パラメータへ写す。
    定数スロットとパラメータを持たないゲートは飛ばすので、評価回数は
    2 * (パラメータを含むスロット数)。

    Args:
        backend: Backend instance used to evaluate the shifted circuits
        circuit: Parameterized circuit
        hamiltonian: Observable
        params: Current parameter values (aligned with circuit.parameters)
        shift: Shift amount (default: π/2)

    Returns:
        Gradient array of shape (len(circuit.parameters),)

    Raises:
        NotImplementedError: for non-linear parameter expressions
            (``a * b``): the shift rule needs the gate angle itself to
            move by exactly s, and there is no constant d(slot)/d(theta)
            to chain-rule back. Adjoint differentiation refuses these
            circuits too.

    Note:
        The 2-term rule assumes the slot enters as exp(-i v G / 2) with
        G^2 = I (rx/ry/rz/u3's angles, p/u1 phases, ...). It is NOT exact
        for the ``r(theta, phi)`` gate's ``phi`` (which sets the rotation
        axis, not an angle conjugate to a two-eigenvalue generator) nor
        for controlled rotations (crx/cry/crz/cp need the 4-term rule).
        Both are pre-existing limitations of this routine; adjoint
        differentiation handles them exactly and is the default.
    """
    gates = _gate_list(circuit)
    bound = circuit.bind_parameters(list(params)) if params is not None \
        else circuit

    nslots = sum(len(g.params) for g in gates)
    slot_grads = np.zeros(nslots)
    factor = 1.0 / (2.0 * np.sin(shift))

    ptr = 0
    for gi, gate in enumerate(gates):
        for si, p in enumerate(gate.params):
            if is_parameterized(p):
                plus = _shifted_circuit(bound, gi, si, shift)
                minus = _shifted_circuit(bound, gi, si, -shift)
                slot_grads[ptr] = factor * (
                    backend.expectation(plus, hamiltonian)
                    - backend.expectation(minus, hamiltonian))
            ptr += 1

    return gate_slot_grads_to_parameter_grads(circuit, slot_grads)
