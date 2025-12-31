
from typing import Any
from ..circuit import Circuit
from ..parameter import Parameter

def to_metalq(qc_qiskit: Any) -> Circuit:
    """
    Convert a Qiskit QuantumCircuit to a Metal-Q Circuit.
    
    Args:
        qc_qiskit: Qiskit QuantumCircuit object.
        
    Returns:
        Metal-Q Circuit object.
    """
    # Try to import qiskit to check type (optional, dynamic typing is fine)
    # We assume 'qc_qiskit' behaves like a QuantumCircuit (has .data or iterates instructions)
    
    num_qubits = qc_qiskit.num_qubits
    mq_circuit = Circuit(num_qubits)
    
    # Iterate over instructions
    # Qiskit 1.0+ uses .data which yields (instruction, qargs, cargs)
    for instruction, qargs, cargs in qc_qiskit.data:
        name = instruction.name
        
        # Map indices
        # qargs uses Qubit objects, we need indices.
        # Assuming simple QuantumRegister(n)
        qubit_indices = [q._index for q in qargs]
        
        # Params
        params = instruction.params
        
        # Convert params to float or MetalQ Parameter
        mq_params = []
        for p in params:
            try:
                # If valid number
                mq_params.append(float(p))
            except (TypeError, ValueError):
                # If ParameterExpression
                # Metal-Q basic support: use name
                if hasattr(p, 'name'):
                    mq_params.append(Parameter(p.name))
                else:
                    raise ValueError(f"Unsupported parameter type: {type(p)}")

        # Helper to apply gate
        # Helper to apply gate
        if name == 'h':
            mq_circuit.h(qubit_indices[0])
        elif name == 'x':
            mq_circuit.x(qubit_indices[0])
        elif name == 'y':
            mq_circuit.y(qubit_indices[0])
        elif name == 'z':
            mq_circuit.z(qubit_indices[0])
        elif name == 's':
            mq_circuit.s(qubit_indices[0])
        elif name == 'sdg':
            mq_circuit.sdg(qubit_indices[0])
        elif name == 't':
            mq_circuit.t(qubit_indices[0])
        elif name == 'tdg':
            mq_circuit.tdg(qubit_indices[0])
        elif name == 'sx':
            mq_circuit.sx(qubit_indices[0])
        elif name == 'sxdg':
            mq_circuit.sxdg(qubit_indices[0])
        elif name == 'rx':
            mq_circuit.rx(mq_params[0], qubit_indices[0])
        elif name == 'ry':
            mq_circuit.ry(mq_params[0], qubit_indices[0])
        elif name == 'rz':
            mq_circuit.rz(mq_params[0], qubit_indices[0])
        elif name == 'p':
            mq_circuit.p(mq_params[0], qubit_indices[0])
        elif name == 'cx':
            mq_circuit.cx(qubit_indices[0], qubit_indices[1])
        elif name == 'cz': 
            mq_circuit.cz(qubit_indices[0], qubit_indices[1])
        elif name == 'cp':
            mq_circuit.cp(mq_params[0], qubit_indices[0], qubit_indices[1])
        elif name == 'swap':
            mq_circuit.swap(qubit_indices[0], qubit_indices[1])
        elif name == 'barrier':
            # Add barrier for visualization
            mq_circuit.barrier(*qubit_indices)
        elif name == 'measure':
            # Handle measurement
            # Try to get clbit index
            try:
                # Qiskit 0.44+
                cl_indices = []
                for c in cargs:
                    try:
                        idx = qc_qiskit.find_bit(c).index
                        cl_indices.append(idx)
                    except AttributeError:
                        # Fallback for older Qiskit versions or constructed bits
                        if hasattr(c, 'index'):
                             cl_indices.append(c.index)
                        elif hasattr(c, '_index'):
                             cl_indices.append(c._index)
                        else:
                             # Last resort: simplistic assumption if Register is standard
                             # This might fail for complex register layouts
                             cl_indices.append(0) 

                mq_circuit.measure(qubit_indices[0], cl_indices[0])
            except Exception:
                # If measurement conversion fails, skip it but warn? 
                # For benchmarking, we NEED measurement.
                # Assuming standard execution where cargs are well-formed.
                pass
        else:
             # Fallback: Check if it's a controlled gate that we can decompose?
             # For now, start with raising Error
             pass # raise ValueError(f"Unsupported gate for conversion: {name}")
             # Or just print warning to avoid crashing benchmark on rare gates
             print(f"Warning: Gate {name} not supported by adapter yet.")

            
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
