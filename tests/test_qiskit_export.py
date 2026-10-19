"""Qiskit export fidelity, without requiring the native Metal library."""

import numpy as np
import pytest

pytest.importorskip("qiskit")
from qiskit.quantum_info import Statevector

from metalq import Circuit, Parameter
from metalq.adapters import to_qiskit
from metalq.backends.cpu.backend import CPUBackend
from metalq.exceptions import ValidationError


@pytest.fixture(scope="module")
def cpu():
    return CPUBackend()


def assert_state_matches(cpu, original, exported):
    np.testing.assert_allclose(
        Statevector.from_instruction(exported).data,
        cpu.statevector(original),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "name,params,wires",
    [
        ("h", [], [1]), ("x", [], [1]), ("y", [], [1]),
        ("z", [], [1]), ("s", [], [1]), ("sdg", [], [1]),
        ("t", [], [1]), ("tdg", [], [1]), ("sx", [], [1]),
        ("sxdg", [], [1]), ("rx", [0.37], [1]),
        ("ry", [-0.52], [1]), ("rz", [0.83], [1]),
        ("p", [0.49], [1]), ("cx", [], [2, 0]),
        ("cz", [], [2, 0]), ("cp", [-0.61], [2, 0]),
        ("swap", [], [2, 0]),
    ],
)
def test_supported_gates_match_original(cpu, name, params, wires):
    original = Circuit(3).ry(0.23, 0).rx(-0.71, 1).h(2)
    getattr(original, name)(*params, *wires)
    original.rz(0.19, 0)
    exported = to_qiskit(original)
    instruction = exported.data[3]
    assert instruction.operation.name == name
    assert instruction.operation.params == params
    assert [exported.find_bit(q).index for q in instruction.qubits] == wires
    assert_state_matches(cpu, original, exported)


@pytest.mark.parametrize("seed", range(5))
def test_mixed_circuits_match_original(cpu, seed):
    rng = np.random.default_rng(seed)
    original = Circuit(5, 0)
    single = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg"]
    rotations = ["rx", "ry", "rz", "p"]
    pairs = ["cx", "cz", "cp", "swap"]
    for _ in range(50):
        name = rng.choice(single + rotations + pairs)
        if name in pairs:
            wires = rng.choice(5, size=2, replace=False).tolist()
        else:
            wires = [int(rng.integers(5))]
        params = (
            [rng.uniform(-np.pi, np.pi)] if name in rotations + ["cp"] else []
        )
        getattr(original, name)(*params, *wires)
    assert_state_matches(cpu, original, to_qiskit(original))


@pytest.mark.parametrize("num_clbits", [0, 1, 3, 5])
def test_classical_bit_count_preserved(num_clbits):
    assert to_qiskit(Circuit(2, num_clbits)).num_clbits == num_clbits


@pytest.mark.parametrize(
    "num_clbits,measurements",
    [(5, [(1, 4), (0, 1)]), (1, [(0, 0), (1, 0)]), (3, [(1, 2), (1, 0)])],
)
def test_terminal_measurement_destinations(num_clbits, measurements):
    original = Circuit(2, num_clbits).h(1).cx(1, 0)
    for q, c in measurements:
        original.measure(q, c)
    exported = to_qiskit(original)
    assert exported.num_clbits == num_clbits
    assert [item.operation.name for item in exported.data[:2]] == ["h", "cx"]
    actual = [
        (exported.find_bit(item.qubits[0]).index,
         exported.find_bit(item.clbits[0]).index)
        for item in exported.data[2:]
    ]
    assert actual == measurements


def test_barrier_preserves_selected_qubit_order():
    exported = to_qiskit(Circuit(3).h(1).barrier(2, 0).x(0))
    barrier = exported.data[1]
    assert barrier.operation.name == "barrier"
    assert [exported.find_bit(q).index for q in barrier.qubits] == [2, 0]


@pytest.mark.parametrize("angle", [-0.8, 0.0, 1.2])
def test_shared_parameter_identity_and_binding(cpu, angle):
    theta = Parameter("theta")
    original = Circuit(2).rx(theta, 0).ry(theta, 1).cp(theta, 1, 0)
    exported = to_qiskit(original)
    assert exported.num_parameters == 1
    parameter = next(iter(exported.parameters))
    assert parameter.name == "theta"
    assert all(item.operation.params[0] == parameter for item in exported.data)
    assert_state_matches(
        cpu, original.bind_parameters({theta: angle}),
        exported.assign_parameters({parameter: angle}),
    )


def test_distinct_parameters_keep_names_and_binding(cpu):
    theta = Parameter("theta[0]")
    phi = Parameter("theta[1]")
    original = Circuit(1).rx(theta, 0).rz(phi, 0).ry(theta, 0)
    exported = to_qiskit(original)
    parameters = {p.name: p for p in exported.parameters}
    assert set(parameters) == {"theta[0]", "theta[1]"}
    assert (
        exported.data[0].operation.params == exported.data[2].operation.params
    )
    bound = exported.assign_parameters(
        {parameters["theta[0]"]: 0.4, parameters["theta[1]"]: -0.2},
    )
    assert_state_matches(
        cpu, original.bind_parameters({theta: 0.4, phi: -0.2}), bound,
    )


def test_distinct_same_named_parameters_rejected():
    first = Parameter("theta")
    second = Parameter("theta")
    original = Circuit(1).rx(first, 0).rz(second, 0)
    with pytest.raises(ValidationError, match="Distinct.*same name.*theta"):
        to_qiskit(original)


def test_same_named_parameters_bound_before_export(cpu):
    first = Parameter("theta")
    second = Parameter("theta")
    original = Circuit(1).rx(first, 0).rz(second, 0)
    bound = original.bind_parameters({first: 0.4, second: -0.2})
    assert_state_matches(cpu, bound, to_qiskit(bound))


@pytest.mark.parametrize("kind", ["scale", "sum", "nested"])
def test_unbound_expressions_rejected_with_binding_guidance(kind):
    theta = Parameter("theta")
    phi = Parameter("phi")
    expression = {
        "scale": 2 * theta, "sum": theta + phi,
        "nested": 2 * (theta + 0.3),
    }[kind]
    with pytest.raises(ValidationError, match="Bind.*before exporting"):
        to_qiskit(Circuit(1).rx(expression, 0))


def test_bound_expression_matches_original(cpu):
    theta = Parameter("theta")
    original = Circuit(1).rx(2 * theta + 0.3, 0)
    bound = original.bind_parameters({theta: 0.4})
    assert_state_matches(cpu, bound, to_qiskit(bound))


@pytest.mark.parametrize("kind", ["cy", "u", "ccx", "id", "custom"])
def test_unsupported_gate_rejected_with_context(kind):
    original = Circuit(3).x(0)
    if kind == "cy":
        original.cy(0, 1)
    elif kind == "u":
        original.u(0.1, 0.2, 0.3, 0)
    elif kind == "ccx":
        original.ccx(0, 1, 2)
    elif kind == "id":
        original.id(0)
    else:
        original._add_gate("custom", [2])
    with pytest.raises(ValidationError) as error:
        to_qiskit(original)
    assert f"'{kind}'" in str(error.value)
    assert "instruction 1" in str(error.value)
    assert f"qubits {original.gates[-1].qubits}" in str(error.value)


@pytest.mark.parametrize(
    "name,wires,params",
    [("rx", [0, 1], []), ("h", [0], [0.7]),
     ("cx", [0, 1, 2], []), ("barrier", [0], [0.5])],
)
def test_malformed_gate_cannot_be_misinterpreted(name, wires, params):
    original = Circuit(3)._add_gate(name, wires, params)
    with pytest.raises(ValidationError, match="Invalid.*instruction 0"):
        to_qiskit(original)


def test_named_nonparameter_is_not_treated_as_a_symbol():
    class NamedValue:
        name = "theta"

    original = Circuit(1).rx(NamedValue(), 0)
    with pytest.raises(ValidationError, match="Unsupported parameter"):
        to_qiskit(original)


@pytest.mark.parametrize("unsupported", [False, True])
def test_input_is_unchanged_on_success_or_failure(unsupported):
    theta = Parameter("theta")
    original = Circuit(2, 3).rx(theta, 1).rz(theta, 0).measure(1, 2)
    if unsupported:
        original.cy(0, 1)
    gates = [(g.name, list(g.qubits), list(g.params)) for g in original.gates]
    measurements = list(original.measurements)
    if unsupported:
        with pytest.raises(ValidationError):
            to_qiskit(original)
    else:
        to_qiskit(original)
    assert gates == [(g.name, g.qubits, g.params) for g in original.gates]
    assert original.measurements == measurements
    assert original.num_clbits == 3
    assert original.parameters == [theta]
