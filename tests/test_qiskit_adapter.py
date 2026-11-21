"""Qiskit import fidelity, independent of the native Metal library."""

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Clbit, Gate, Parameter, ParameterVector, Qubit
from qiskit.circuit.library import U3Gate
from qiskit.quantum_info import Statevector

from metalq import Circuit
from metalq import Parameter as MetalQParameter
from metalq.adapters.qiskit_adapter import to_metalq, to_qiskit
from metalq.backends.cpu.backend import CPUBackend
from metalq.exceptions import ValidationError

# (qubit count, parameter count) of every gate that to_qiskit exports.
GATES = {
    "id": (1, 0), "h": (1, 0), "x": (1, 0), "y": (1, 0), "z": (1, 0),
    "s": (1, 0), "sdg": (1, 0), "t": (1, 0), "tdg": (1, 0), "sx": (1, 0),
    "sxdg": (1, 0), "rx": (1, 1), "ry": (1, 1), "rz": (1, 1), "p": (1, 1),
    "r": (1, 2), "u": (1, 3), "cx": (2, 0), "cy": (2, 0), "cz": (2, 0),
    "ch": (2, 0), "cp": (2, 1), "crx": (2, 1), "cry": (2, 1),
    "crz": (2, 1), "swap": (2, 0), "iswap": (2, 0), "rxx": (2, 1),
    "ryy": (2, 1), "rzx": (2, 1), "rzz": (2, 1), "ccx": (3, 0),
    "ccz": (3, 0), "cswap": (3, 0),
}
# Non-adjacent and reversed wires on four qubits, by gate arity.
WIRE_ORDERS = {
    1: [[3], [1]],
    2: [[3, 0], [0, 2], [2, 1]],
    3: [[3, 0, 2], [0, 3, 1], [2, 1, 0]],
}
GATE_CASES = [
    (name, wires) for name, (arity, _) in GATES.items()
    for wires in WIRE_ORDERS[arity]
]
SHARED = object()  # Placeholder for the shared parameter in round trips.


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


def entangled_circuit(rng, num_qubits=4):
    """Prepare a generic entangled state from supported gates."""
    qc = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        qc.ry(rng.uniform(-np.pi, np.pi), q)
        qc.rz(rng.uniform(-np.pi, np.pi), q)
    for q in range(num_qubits - 1):
        qc.cx(q, q + 1)
    for q in range(num_qubits):
        qc.rx(rng.uniform(-np.pi, np.pi), q)
    return qc


@pytest.mark.parametrize(
    "name,params,wires",
    [
        ("id", [], [1]), ("h", [], [1]), ("x", [], [1]), ("y", [], [1]),
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


@pytest.mark.parametrize(
    "case,name,wires",
    [(i, name, wires) for i, (name, wires) in enumerate(GATE_CASES)],
    ids=[f"{name}-{wires}" for name, wires in GATE_CASES],
)
def test_all_exported_gates_match_qiskit_on_entangled_states(
        cpu, case, name, wires):
    rng = np.random.default_rng(case)
    qc = entangled_circuit(rng)
    position = len(qc.data)
    params = rng.uniform(-np.pi, np.pi, GATES[name][1]).tolist()
    getattr(qc, name)(*params, *wires)
    converted = to_metalq(qc)
    gate = converted.gates[position]
    assert (gate.name, gate.qubits, gate.params) == (name, wires, params)
    assert_state_matches(cpu, converted, qc)


def test_bell_state(cpu):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    assert_state_matches(cpu, to_metalq(qc), qc)


def test_transpiled_bell_state_including_global_phase(cpu):
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    compiled = qiskit.transpile(
        bell, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1,
    )
    assert compiled.global_phase != 0
    assert_state_matches(cpu, to_metalq(compiled), compiled)


@pytest.mark.parametrize("seed", range(5))
def test_mixed_circuits_match_qiskit(cpu, seed):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(QuantumRegister(2, "a"), QuantumRegister(3, "b"))
    for _ in range(50):
        name = str(rng.choice(list(GATES)))
        num_qubits, num_params = GATES[name]
        wires = rng.choice(5, size=num_qubits, replace=False).tolist()
        params = rng.uniform(-np.pi, np.pi, num_params).tolist()
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
    "kind",
    ["cs", "u3", "reset", "custom", "if_else", "open_cx", "open_crx",
     "open_ccx"],
)
def test_unsupported_operations_rejected_with_context(kind):
    qc = QuantumCircuit(3, 1)
    qc.x(0)
    if kind == "cs":
        qc.cs(0, 1)
    elif kind == "u3":
        qc.append(U3Gate(0.3, -0.2, 0.5), [1])
    elif kind == "open_crx":
        qc.crx(np.pi / 3, 2, 0, ctrl_state=0)
    elif kind == "open_ccx":
        qc.ccx(0, 2, 1, ctrl_state=1)
    elif kind == "reset":
        qc.reset(0)
    elif kind == "custom":
        qc.append(Gate("custom", 1, []), [0])
    elif kind == "if_else":
        with qc.if_test((qc.clbits[0], True)):
            qc.x(0)
    else:
        qc.cx(0, 1, ctrl_state=0)
    with pytest.raises(ValidationError, match=r"instruction 1.*qubits \["):
        to_metalq(qc)


@pytest.mark.parametrize(
    "name,num_qubits,params",
    [("x", 1, []), ("delay", 1, []), ("id", 1, []),
     ("u", 1, [0.1, 0.2, 0.3]), ("rzz", 2, [0.4]), ("cswap", 3, [])],
)
def test_custom_gate_cannot_impersonate_supported_name(
        name, num_qubits, params):
    qc = QuantumCircuit(num_qubits)
    custom = Gate(name, num_qubits, params)
    definition = QuantumCircuit(num_qubits)
    definition.h(range(num_qubits))
    custom.definition = definition
    qc.append(custom, range(num_qubits))
    with pytest.raises(ValidationError, match=f"Unsupported.*'{name}'"):
        to_metalq(qc)


def test_classically_conditioned_standard_gate_rejected():
    qc = QuantumCircuit(1, 1)
    conditional_x = qiskit.circuit.library.XGate().to_mutable()
    conditional_x.condition = (qc.clbits[0], 1)
    qc.append(conditional_x, [0])
    with pytest.raises(ValidationError, match="Classically conditioned"):
        to_metalq(qc)


@pytest.mark.parametrize("name,wires", [("x", [0]), ("cx", [0, 1]), ("cx", [1, 0])])
def test_gate_on_measured_qubit_rejected(name, wires):
    qc = QuantumCircuit(2, 1)
    qc.measure(0, 0)
    qc.barrier()
    getattr(qc, name)(*wires)
    with pytest.raises(ValidationError, match="terminal measurements"):
        to_metalq(qc)


def test_independent_gates_and_delays_after_measurement(cpu):
    qc = QuantumCircuit(2, 2, global_phase=0.4)
    qc.h(0)
    qc.delay(10, 0, unit="dt")
    qc.measure(0, 1)
    qc.delay(20, 0, unit="ns")
    qc.h(1)
    qc.measure(1, 0)
    converted = to_metalq(qc)
    assert converted.measurements == [(0, 1), (1, 0)]
    assert "delay" not in [gate.name for gate in converted.gates]
    reference = QuantumCircuit(2, global_phase=0.4)
    reference.h([0, 1])
    assert_state_matches(cpu, converted, reference)


@pytest.mark.parametrize("phase", [0.3, -0.7, np.pi])
def test_numeric_global_phase_preserved(cpu, phase):
    qc = QuantumCircuit(2, global_phase=phase)
    qc.h(0)
    qc.cx(0, 1)
    assert_state_matches(cpu, to_metalq(qc), qc)


@pytest.mark.parametrize("phase", [0, 2 * np.pi, -1e-17, 1e-17])
def test_trivial_global_phase_adds_no_gates(cpu, phase):
    qc = QuantumCircuit(1, global_phase=phase)
    converted = to_metalq(qc)
    assert converted.gates == []
    assert_state_matches(cpu, converted, qc)


def test_symbolic_global_phase_requires_binding(cpu):
    phase = Parameter("phase")
    qc = QuantumCircuit(1, global_phase=2 * phase)
    with pytest.raises(ValidationError, match="global phase"):
        to_metalq(qc)
    bound = qc.assign_parameters({phase: 0.3})
    assert_state_matches(cpu, to_metalq(bound), bound)


@pytest.mark.parametrize("unsupported", [False, True])
def test_conversion_does_not_mutate_input(unsupported):
    theta = Parameter("theta")
    qc = QuantumCircuit(2, 3)
    qc.rx(theta, 1)
    if unsupported:
        qc.cs(0, 1)
    qc.measure(1, 2)
    original = qc.copy()
    if unsupported:
        with pytest.raises(ValidationError):
            to_metalq(qc)
    else:
        to_metalq(qc)
    assert qc == original
    assert qc.num_parameters == 1


def round_trip_spec():
    """Every exported gate on shuffled wires, as (name, params, wires).

    Each parameter slot of each parameterized gate holds SHARED once, with
    distinct numbers elsewhere, so permuted slots or lost sharing show up.
    """
    rng = np.random.default_rng(7)
    gates = []
    for name, (num_qubits, num_params) in GATES.items():
        for slot in range(max(num_params, 1)):
            params = rng.uniform(-np.pi, np.pi, num_params).tolist()
            if num_params:
                params[slot] = SHARED
            wires = rng.choice(4, size=num_qubits, replace=False).tolist()
            gates.append((name, params, wires))
    order = rng.permutation(len(gates))
    prep = [("ry", [rng.uniform(-np.pi, np.pi)], [q]) for q in range(4)]
    return prep + [gates[i] for i in order]


def build_round_trip_circuit(circuit, spec, parameter):
    for name, params, wires in spec:
        values = [parameter if p is SHARED else p for p in params]
        getattr(circuit, name)(*values, *wires)
    return circuit


def expected_gates(spec, parameter):
    return [
        (name, wires, [parameter if p is SHARED else p for p in params])
        for name, params, wires in spec
    ]


@pytest.mark.parametrize("angle", [-0.8, 1.3])
def test_metalq_to_qiskit_to_metalq_round_trip(cpu, angle):
    spec = round_trip_spec()
    assert {name for name, _, _ in spec} == set(GATES)
    theta = MetalQParameter("theta")
    original = build_round_trip_circuit(Circuit(4), spec, theta)
    exported = to_qiskit(original)
    restored = to_metalq(exported)

    assert exported.num_parameters == 1
    assert restored.num_parameters == 1
    shared = restored.parameters[0]
    assert shared is not theta and shared.name == "theta"
    actual = [(g.name, g.qubits, g.params) for g in restored.gates]
    assert actual == expected_gates(spec, shared)
    assert all(
        p is shared for g in restored.gates for p in g.params
        if isinstance(p, MetalQParameter)
    )

    bound = restored.bind_parameters({shared: angle})
    np.testing.assert_allclose(
        cpu.statevector(bound),
        cpu.statevector(original.bind_parameters({theta: angle})),
        rtol=1e-12, atol=1e-12,
    )
    qiskit_parameter = next(iter(exported.parameters))
    assert_state_matches(
        cpu, bound, exported.assign_parameters({qiskit_parameter: angle}),
    )


@pytest.mark.parametrize("angle", [-0.8, 1.3])
def test_qiskit_to_metalq_to_qiskit_round_trip(cpu, angle):
    spec = round_trip_spec()
    theta = Parameter("theta")
    original = build_round_trip_circuit(QuantumCircuit(4), spec, theta)
    imported = to_metalq(original)
    restored = to_qiskit(imported)

    assert imported.num_parameters == 1
    imported_theta = imported.parameters[0]
    assert imported_theta.name == "theta"
    assert all(
        p is imported_theta for g in imported.gates for p in g.params
        if isinstance(p, MetalQParameter)
    )
    assert restored.num_parameters == 1
    shared = next(iter(restored.parameters))
    assert shared.name == "theta"
    actual = [
        (item.operation.name,
         [restored.find_bit(q).index for q in item.qubits],
         list(item.operation.params))
        for item in restored.data
    ]
    assert actual == expected_gates(spec, shared)

    reference = original.assign_parameters({theta: angle})
    np.testing.assert_allclose(
        Statevector(restored.assign_parameters({shared: angle})).data,
        Statevector(reference).data,
        rtol=1e-12, atol=1e-12,
    )
    assert_state_matches(
        cpu, imported.bind_parameters({imported_theta: angle}), reference,
    )
