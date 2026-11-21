"""Conversion between Qiskit and Metal-Q circuits."""

import math
from typing import Any

from ..circuit import Circuit
from ..exceptions import ValidationError
from ..parameter import Parameter


def to_metalq(qc_qiskit: Any) -> Circuit:
    """Convert a Qiskit QuantumCircuit without silently changing its meaning.

    Supported standard gates are id, h, x, y, z, s, sdg, t, tdg, sx, sxdg,
    rx, ry, rz, p, r and u; cx, cy, cz, ch, cp, crx, cry, crz, swap, iswap,
    rxx, ryy, rzx and rzz; and ccx, ccz and cswap, matching to_qiskit.
    Open-controlled variants (such as ctrl_state=0) are rejected. Numeric
    global phases are encoded using rz and p. Delays are ignored in ideal
    simulation. Barriers and measurements terminal on their qubits are
    supported across registers and loose bits.
    Classical bit counts and measurement destinations are preserved. This
    does not change the backend's counts formatting or sampling behavior.

    Repeated Qiskit Parameters (including ParameterVector elements) map to
    the same Metal-Q Parameter. Bind general ParameterExpressions in Qiskit
    before conversion. The input circuit is never modified or transpiled.

    Args:
        qc_qiskit: Qiskit QuantumCircuit object.

    Returns:
        Metal-Q Circuit with the same supported gates and measurement map.

    Raises:
        ValidationError: The circuit contains an unsupported operation,
            classical condition, gate on a previously measured qubit, unbound
            parameter expression, or unbound global phase. Instruction indices
            in error messages are zero-based.
        ImportError: Qiskit is not installed.
    """
    from qiskit.circuit import Barrier, Delay, Measure
    from qiskit.circuit import Parameter as QiskitParameter
    from qiskit.circuit import library

    # Names alone are insufficient: a custom gate named "x" may implement H.
    # base_class also handles Qiskit's singleton standard gates.
    supported_types = {
        'h': library.HGate, 'x': library.XGate, 'y': library.YGate,
        'z': library.ZGate, 's': library.SGate, 'sdg': library.SdgGate,
        't': library.TGate, 'tdg': library.TdgGate, 'sx': library.SXGate,
        'sxdg': library.SXdgGate, 'rx': library.RXGate,
        'ry': library.RYGate, 'rz': library.RZGate, 'p': library.PhaseGate,
        'r': library.RGate, 'u': library.UGate,
        'cx': library.CXGate, 'cy': library.CYGate, 'cz': library.CZGate,
        'ch': library.CHGate, 'cp': library.CPhaseGate,
        'crx': library.CRXGate, 'cry': library.CRYGate,
        'crz': library.CRZGate, 'swap': library.SwapGate,
        'iswap': library.iSwapGate, 'rxx': library.RXXGate,
        'ryy': library.RYYGate, 'rzx': library.RZXGate,
        'rzz': library.RZZGate, 'ccx': library.CCXGate,
        'ccz': library.CCZGate, 'cswap': library.CSwapGate,
        'id': library.IGate, 'barrier': Barrier, 'measure': Measure,
        'delay': Delay,
    }

    phase = qc_qiskit.global_phase
    if getattr(phase, 'parameters', ()):
        raise ValidationError(
            "Bind the symbolic global phase in Qiskit before converting."
        )

    mq_circuit = Circuit(qc_qiskit.num_qubits, qc_qiskit.num_clbits)
    phase = math.remainder(float(phase), math.tau)
    if abs(phase) > 1e-15:
        # P(2φ) RZ(-2φ) = exp(iφ) I on either basis state.
        mq_circuit.rz(-2 * phase, 0)
        mq_circuit.p(2 * phase, 0)
    parameter_map = {}
    measured_qubits = set()

    for index, item in enumerate(qc_qiskit.data):
        operation = item.operation
        name = operation.name
        qubits = [qc_qiskit.find_bit(q).index for q in item.qubits]
        context = f"'{name}' at instruction {index} on qubits {qubits}"

        if getattr(operation, 'condition', None) is not None:
            raise ValidationError(
                f"Classically conditioned operation {context} is unsupported."
            )
        if (name not in supported_types
                or operation.base_class is not supported_types[name]):
            raise ValidationError(f"Unsupported operation {context}.")
        if name == 'delay':
            continue
        if name == 'measure':
            clbit = qc_qiskit.find_bit(item.clbits[0]).index
            mq_circuit.measure(qubits[0], clbit)
            measured_qubits.add(qubits[0])
            continue
        if name != 'barrier' and not measured_qubits.isdisjoint(qubits):
            raise ValidationError(
                f"Cannot convert {context}: only terminal measurements "
                "are supported (gate acts on a previously measured qubit)."
            )

        params = []
        for value in operation.params:
            if isinstance(value, QiskitParameter):
                # Qiskit Parameter equality/hash use UUID, not just the name.
                if value not in parameter_map:
                    parameter_map[value] = Parameter(value.name)
                params.append(parameter_map[value])
            else:
                message = (
                    f"Unsupported parameter in {context}. Bind parameter "
                    "expressions in Qiskit before converting."
                )
                if getattr(value, 'parameters', ()):
                    raise ValidationError(message)
                try:
                    params.append(float(value))
                except (TypeError, ValueError) as exc:
                    raise ValidationError(message) from exc

        getattr(mq_circuit, name)(*params, *qubits)

    return mq_circuit


def to_qiskit(mq_circuit: Circuit) -> Any:
    """Export a Metal-Q Circuit without silently dropping gates or symbols.

    Measurement-free circuits export with zero classical bits so that Qiskit
    can use them as gates. For measured circuits, the full classical register
    is preserved and measurements are appended after all gates.

    Repeated Metal-Q Parameters map to one Qiskit Parameter per identity.
    Distinct parameters with the same name are rejected because Qiskit
    requires unique parameter names within a circuit. Bind them or give them
    unique names before exporting. Bind general ParameterExpressions using
    Circuit.bind_parameters before exporting. The input is never modified.

    Args:
        mq_circuit: Metal-Q Circuit object.

    Returns:
        Qiskit QuantumCircuit with the supported gates and measurement map.

    Raises:
        ValidationError: A gate is unsupported or has an invalid shape, a
            parameter value cannot be exported, or distinct parameters share
            a name. Instruction indices in error messages are zero-based.
        ImportError: Qiskit is not installed.
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter as QiskitParameter
    except ImportError as exc:
        raise ImportError(
            "Qiskit is not installed. Install 'qiskit' to use this feature."
        ) from exc

    from ..parameter import ParameterExpression

    # (qubit count, parameter count); barriers take any number of qubits.
    supported = {
        'h': (1, 0), 'x': (1, 0), 'y': (1, 0), 'z': (1, 0),
        's': (1, 0), 'sdg': (1, 0), 't': (1, 0), 'tdg': (1, 0),
        'sx': (1, 0), 'sxdg': (1, 0), 'rx': (1, 1), 'ry': (1, 1),
        'rz': (1, 1), 'p': (1, 1), 'cx': (2, 0), 'cz': (2, 0),
        'cp': (2, 1), 'swap': (2, 0), 'id': (1, 0), 'u': (1, 3),
        'r': (1, 2), 'cy': (2, 0), 'ch': (2, 0), 'iswap': (2, 0),
        'crx': (2, 1), 'cry': (2, 1), 'crz': (2, 1), 'rxx': (2, 1),
        'ryy': (2, 1), 'rzx': (2, 1), 'rzz': (2, 1), 'ccx': (3, 0),
        'ccz': (3, 0), 'cswap': (3, 0), 'barrier': (None, 0),
    }
    measurements = mq_circuit.measurements
    num_clbits = mq_circuit.num_clbits if measurements else 0
    qc = QuantumCircuit(mq_circuit.num_qubits, num_clbits)
    parameter_map = {}
    parameter_names = set()

    for index, gate in enumerate(mq_circuit.gates):
        name = gate.name.lower()
        context = f"'{name}' at instruction {index} on qubits {gate.qubits}"
        if name not in supported:
            raise ValidationError(f"Unsupported gate {context}.")
        num_qubits, num_params = supported[name]
        if ((num_qubits is not None and len(gate.qubits) != num_qubits)
                or len(gate.params) != num_params):
            raise ValidationError(f"Invalid gate shape for {context}.")

        params = []
        for value in gate.params:
            if isinstance(value, Parameter):
                if value not in parameter_map:
                    if value.name in parameter_names:
                        raise ValidationError(
                            f"Distinct parameters have the same name "
                            f"'{value.name}' in {context}. Bind them or give "
                            "them unique names before exporting."
                        )
                    parameter_map[value] = QiskitParameter(value.name)
                    parameter_names.add(value.name)
                params.append(parameter_map[value])
            elif isinstance(value, ParameterExpression):
                raise ValidationError(
                    f"Unsupported parameter expression in {context}. Bind "
                    "expressions with Circuit.bind_parameters before "
                    "exporting."
                )
            else:
                try:
                    params.append(float(value))
                except (TypeError, ValueError) as exc:
                    raise ValidationError(
                        f"Unsupported parameter in {context}. Expected a "
                        "numeric value or Metal-Q Parameter."
                    ) from exc

        getattr(qc, name)(*params, *gate.qubits)

    for qubit, clbit in measurements:
        qc.measure(qubit, clbit)

    return qc
