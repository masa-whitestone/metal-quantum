"""Qiskit import fidelity, independent of the native Metal library."""

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Clbit, Gate, Parameter, ParameterVector, Qubit
from qiskit.quantum_info import Statevector

from metalq.adapters.qiskit_adapter import to_metalq
from metalq.backends.cpu.backend import CPUBackend
from metalq.exceptions import ValidationError


@pytest.fixture(scope="module")
def cpu():
    return CPUBackend()


def assert_state_matches(cpu, converted, reference):
    np.testing.assert_allclose(
        cpu.statevector(converted),
        Statevector.from_instruction(reference).data,
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
def test_supported_gates_match_qiskit(cpu, name, params, wires):
    qc = QuantumCircuit(3)
    qc.ry(0.23, 0)
    qc.rx(-0.71, 1)
    qc.h(2)
    getattr(qc, name)(*params, *wires)
    qc.rz(0.19, 0)
    converted = to_metalq(qc)
    assert converted.gates[3].name == name
    assert converted.gates[3].qubits == wires
    assert converted.gates[3].params == params
    assert_state_matches(cpu, converted, qc)


def test_bell_state(cpu):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    assert_state_matches(cpu, to_metalq(qc), qc)


@pytest.mark.parametrize("seed", range(5))
def test_mixed_circuits_match_qiskit(cpu, seed):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(QuantumRegister(2, "a"), QuantumRegister(3, "b"))
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
        getattr(qc, name)(*params, *wires)
    assert_state_matches(cpu, to_metalq(qc), qc)


@pytest.mark.parametrize(
    "name,wires",
    [("h", [2]), ("cx", [2, 0]), ("cx", [0, 2]), ("swap", [2, 1])],
)
def test_multiple_quantum_registers(cpu, name, wires):
    a = QuantumRegister(2, "a")
    b = QuantumRegister(1, "b")
    qc = QuantumCircuit(a, b)
    qc.ry(0.45, a[0])
    qc.rx(-0.32, b[0])
    getattr(qc, name)(*(qc.qubits[i] for i in wires))
    converted = to_metalq(qc)
    assert converted.gates[-1].qubits == wires
    assert_state_matches(cpu, converted, qc)


def test_loose_bits_and_terminal_measurement():
    qc = QuantumCircuit([Qubit(), Qubit(), Clbit(), Clbit(), Clbit()])
    qc.h(qc.qubits[1])
    qc.measure(qc.qubits[1], qc.clbits[2])
    converted = to_metalq(qc)
    assert converted.gates[0].qubits == [1]
    assert converted.num_clbits == 3
    assert converted.measurements == [(1, 2)]


@pytest.mark.parametrize("n_clbits", [0, 1, 3, 5])
def test_classical_bit_count_preserved(n_clbits):
    assert to_metalq(QuantumCircuit(2, n_clbits)).num_clbits == n_clbits


def test_multiple_classical_registers_and_nonsequential_measurements():
    a = QuantumRegister(1, "a")
    b = QuantumRegister(1, "b")
    c = ClassicalRegister(2, "c")
    d = ClassicalRegister(3, "d")
    qc = QuantumCircuit(a, b, c, d)
    qc.h(b[0])
    qc.measure(b[0], d[2])
    qc.measure(a[0], c[1])
    qc.barrier()
    converted = to_metalq(qc)
    assert converted.num_qubits == 2
    assert converted.num_clbits == 5
    assert converted.measurements == [(1, 4), (0, 1)]


def test_measurement_errors_are_not_swallowed(monkeypatch):
    def reject_measurement(self, qubit, clbit):
        raise ValueError("measurement rejected")

    monkeypatch.setattr(
        "metalq.adapters.qiskit_adapter.Circuit.measure", reject_measurement,
    )
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    with pytest.raises(ValueError, match="measurement rejected"):
        to_metalq(qc)


def test_barrier_preserves_global_qubit_indices():
    a = QuantumRegister(1, "a")
    b = QuantumRegister(2, "b")
    qc = QuantumCircuit(a, b)
    qc.barrier(b[1], a[0])
    assert to_metalq(qc).gates[0].qubits == [2, 0]


@pytest.mark.parametrize("angle", [-0.8, 0.0, 1.2])
def test_shared_parameter_identity_and_binding(cpu, angle):
    theta = Parameter("theta")
    qc = QuantumCircuit(2)
    qc.rx(theta, 0)
    qc.ry(theta, 1)
    qc.cp(theta, 1, 0)
    converted = to_metalq(qc)
    assert converted.num_parameters == 1
    assert all(g.params[0] is converted.parameters[0] for g in converted.gates)
    bound = converted.bind_parameters({converted.parameters[0]: angle})
    assert_state_matches(cpu, bound, qc.assign_parameters({theta: angle}))


def test_parameter_vector_elements_remain_distinct(cpu):
    theta = ParameterVector("theta", 2)
    qc = QuantumCircuit(1)
    qc.rx(theta[0], 0)
    qc.rz(theta[1], 0)
    qc.ry(theta[0], 0)
    converted = to_metalq(qc)
    assert converted.num_parameters == 2
    assert converted.gates[0].params[0] is converted.gates[2].params[0]
    assert converted.gates[0].params[0] is not converted.gates[1].params[0]
    values = {p: {"theta[0]": 0.4, "theta[1]": -0.2}[p.name]
              for p in converted.parameters}
    reference = qc.assign_parameters({theta[0]: 0.4, theta[1]: -0.2})
    assert_state_matches(cpu, converted.bind_parameters(values), reference)


@pytest.mark.parametrize(
    "kind", ["scale", "offset", "sum", "sin", "cancelled"],
)
def test_unbound_parameter_expressions_rejected(kind):
    theta = Parameter("theta")
    phi = Parameter("phi")
    expression = {"scale": 2 * theta, "offset": theta + 0.3,
                  "sum": theta + phi, "sin": theta.sin(),
                  "cancelled": theta - theta}[kind]
    qc = QuantumCircuit(1)
    qc.rx(expression, 0)
    with pytest.raises(
        ValidationError, match="Bind.*Qiskit.*before converting",
    ):
        to_metalq(qc)


def test_expression_bound_in_qiskit_is_supported(cpu):
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.rx(2 * theta + 0.3, 0)
    bound = qc.assign_parameters({theta: 0.4})
    assert_state_matches(cpu, to_metalq(bound), bound)


@pytest.mark.parametrize(
    "kind", ["cry", "reset", "delay", "custom", "if_else", "open_cx"],
)
def test_unsupported_operations_rejected_with_context(kind):
    qc = QuantumCircuit(2, 1)
    qc.x(0)
    if kind == "cry":
        qc.cry(np.pi / 3, 0, 1)
    elif kind == "reset":
        qc.reset(0)
    elif kind == "delay":
        qc.delay(10, 0, unit="dt")
    elif kind == "custom":
        qc.append(Gate("custom", 1, []), [0])
    elif kind == "if_else":
        with qc.if_test((qc.clbits[0], True)):
            qc.x(0)
    else:
        qc.cx(0, 1, ctrl_state=0)
    with pytest.raises(ValidationError, match=r"instruction 1.*qubits \["):
        to_metalq(qc)


def test_custom_gate_cannot_impersonate_supported_name():
    qc = QuantumCircuit(1)
    custom = Gate("x", 1, [])
    definition = QuantumCircuit(1)
    definition.h(0)
    custom.definition = definition
    qc.append(custom, [0])
    with pytest.raises(ValidationError, match="Unsupported.*'x'"):
        to_metalq(qc)


def test_classically_conditioned_standard_gate_rejected():
    qc = QuantumCircuit(1, 1)
    conditional_x = qiskit.circuit.library.XGate().to_mutable()
    conditional_x.condition = (qc.clbits[0], 1)
    qc.append(conditional_x, [0])
    with pytest.raises(ValidationError, match="Classically conditioned"):
        to_metalq(qc)


@pytest.mark.parametrize("same_qubit", [True, False])
def test_gate_after_measurement_rejected(same_qubit):
    qc = QuantumCircuit(2, 1)
    qc.measure(0, 0)
    qc.barrier()
    qc.x(0 if same_qubit else 1)
    with pytest.raises(ValidationError, match="terminal measurements"):
        to_metalq(qc)


@pytest.mark.parametrize("phase", [0.3, np.pi, Parameter("phase")])
def test_nonzero_or_symbolic_global_phase_rejected(phase):
    qc = QuantumCircuit(1, global_phase=phase)
    with pytest.raises(ValidationError, match="global phase"):
        to_metalq(qc)


@pytest.mark.parametrize("unsupported", [False, True])
def test_conversion_does_not_mutate_input(unsupported):
    theta = Parameter("theta")
    qc = QuantumCircuit(2, 3)
    qc.rx(theta, 1)
    if unsupported:
        qc.cry(0.2, 0, 1)
    qc.measure(1, 2)
    original = qc.copy()
    if unsupported:
        with pytest.raises(ValidationError):
            to_metalq(qc)
    else:
        to_metalq(qc)
    assert qc == original
    assert qc.num_parameters == 1
