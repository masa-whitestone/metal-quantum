"""Conversion between Qiskit and Metal-Q circuits."""

from typing import Any

from ..circuit import Circuit
from ..exceptions import ValidationError
from ..parameter import Parameter


def to_metalq(qc_qiskit: Any) -> Circuit:
    """Convert a Qiskit QuantumCircuit without silently changing its meaning.

    Supported standard gates are h, x, y, z, s, sdg, t, tdg, sx, sxdg,
    rx, ry, rz, p, cx, cz, cp and swap. Barriers and terminal measurements
    are supported, including circuits with multiple registers or loose bits.
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
            classical condition, gate after a measurement, unbound parameter
            expression, or nonzero/symbolic global phase. Instruction indices
            in error messages are zero-based.
        ImportError: Qiskit is not installed.
    """
    from qiskit.circuit import Barrier, Measure
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
        'cx': library.CXGate, 'cz': library.CZGate,
        'cp': library.CPhaseGate, 'swap': library.SwapGate,
        'barrier': Barrier, 'measure': Measure,
    }

    phase = qc_qiskit.global_phase
    if getattr(phase, 'parameters', ()) or float(phase) != 0.0:
        raise ValidationError(
            "Cannot convert a nonzero or symbolic global phase: "
            "Metal-Q Circuit has no global phase representation."
        )

    mq_circuit = Circuit(qc_qiskit.num_qubits, qc_qiskit.num_clbits)
    parameter_map = {}
    seen_measurement = False

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
        if seen_measurement and name not in ('measure', 'barrier'):
            raise ValidationError(
                f"Cannot convert {context}: only terminal measurements "
                "are supported (no gates may follow a measurement)."
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

        if name == 'measure':
            clbit = qc_qiskit.find_bit(item.clbits[0]).index
            mq_circuit.measure(qubits[0], clbit)
            seen_measurement = True
        else:
            getattr(mq_circuit, name)(*params, *qubits)

    return mq_circuit


def to_qiskit(mq_circuit: Circuit) -> Any:
    """
    Convert a Metal-Q Circuit to a Qiskit QuantumCircuit.
    
    Args:
        mq_circuit: Metal-Q Circuit object.
        
    Returns:
        Qiskit QuantumCircuit object.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        raise ImportError("Qiskit is not installed. Please install 'qiskit' to use this feature.")
        
    qc = QuantumCircuit(mq_circuit.num_qubits)
    
    for gate in mq_circuit.gates:
        name = gate.name.lower()
        qubits = gate.qubits
        params = gate.params
        
        # Resolve parameters if they are floats, keep Expressions/Parameters if possible?
        # Qiskit supports parameters. We should map MetalQ Parameter -> Qiskit Parameter?
        # For simplicity in V1, we assume values or simple mapping if needed.
        # But Qiskit Parameter objects are distinct.
        # If params contain MetalQ Parameters, we should probably convert them to Qiskit Parameters.
        # This requires maintaining a map of name -> QiskitParameter.
        
        # Simple mapping for now (Values only or simple casting)
        # TODO: Full Parameter mapping
        
        q_params = []
        for p in params:
             if hasattr(p, 'name'):
                 from qiskit.circuit import Parameter as QParameter
                 # Note: This creates a NEW parameter with same name.
                 # If reused, Qiskit treats them as same.
                 q_params.append(QParameter(p.name))
             else:
                 q_params.append(float(p))
        
        if name == 'h':
            qc.h(qubits[0])
        elif name == 'x':
            qc.x(qubits[0])
        elif name == 'y':
            qc.y(qubits[0])
        elif name == 'z':
            qc.z(qubits[0])
        elif name == 'rx':
            qc.rx(q_params[0], qubits[0])
        elif name == 'ry':
            qc.ry(q_params[0], qubits[0])
        elif name == 'rz':
            qc.rz(q_params[0], qubits[0])
        elif name == 'cx':
            qc.cx(qubits[0], qubits[1])
        elif name == 'cz':
            qc.cz(qubits[0], qubits[1])
        elif name == 'swap':
            qc.swap(qubits[0], qubits[1])
        elif name == 'barrier':
            qc.barrier(qubits)
        else:
            # Fallback or warning
            print(f"Warning: Gate {name} not directly mapped to Qiskit. Skipping.")
            
    # Measurements? Metal-Q stores measurements separately
    for q_idx, c_idx in mq_circuit.measurements:
        qc.measure(q_idx, c_idx)
        
    return qc
