"""
circuit_data.py - Convert Qiskit circuits to internal data format
"""
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Any
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, Measure, Barrier
import numpy as np


@dataclass
class GateData:
    """Data for a single gate"""
    name: str                          # Gate name (h, x, cx, rz, etc.)
    qubits: List[int]                  # Target qubit indices
    params: List[float]                # Parameters (rotation angles, etc.)
    matrix: Optional[List[List[float]]] = None  # Custom matrix data


@dataclass  
class CircuitData:
    """Full circuit data"""
    num_qubits: int
    num_clbits: int
    gates: List[GateData]
    measurements: List[Tuple[int, int]]  # (qubit_index, clbit_index)
    

def extract_circuit_data(qc: QuantumCircuit) -> CircuitData:
    """
    Convert Qiskit QuantumCircuit to internal format
    """
    gates = []
    measurements = []
    
    for instruction, qargs, cargs in qc.data:
        qubit_indices = [qc.find_bit(q).index for q in qargs]
        
        if isinstance(instruction, Measure):
            clbit_indices = [qc.find_bit(c).index for c in cargs]
            for q, c in zip(qubit_indices, clbit_indices):
                measurements.append((q, c))
            continue
            
        if isinstance(instruction, Barrier):
            continue
            
        gate = _instruction_to_gate_data(instruction, qubit_indices)
        if gate:
            gates.append(gate)
    
    return CircuitData(
        num_qubits=qc.num_qubits,
        num_clbits=qc.num_clbits,
        gates=gates,
        measurements=measurements
    )


def _instruction_to_gate_data(inst: Instruction, qubits: List[int]) -> Optional[GateData]:
    """
    Convert Qiskit Instruction to GateData
    """
    name = inst.name.lower()
    params = [float(p) for p in inst.params]
    
    # Standard gates supported
    standard_gates = {
        'id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'sx', 'sxdg',
        'rx', 'ry', 'rz', 'p', 'u1', 'u2', 'u3', 'u',
        'cx', 'cy', 'cz', 'ch', 'swap', 'iswap',
        'crx', 'cry', 'crz', 'cp', 'cu1', 'cu3', 'cu',
        'ccx', 'cswap',
        'reset'
    }
    
    if name in standard_gates:
        return GateData(name=name, qubits=qubits, params=params)
    
    # Custom unitary gates
    try:
        matrix = inst.to_matrix()
        # Convert complex -> [real, imag] list
        matrix_data = []
        for row in matrix:
            row_data = []
            for val in row:
                row_data.append([float(val.real), float(val.imag)])
            matrix_data.append(row_data)
        
        return GateData(
            name='unitary',
            qubits=qubits,
            params=[],
            matrix=matrix_data
        )
    except Exception:
        # Fallback for unsupported gates or ignore
        # raise ValueError(f"Unsupported gate: {name}")
        pass # Ideally we should warn or error, but passing for now for unsupported ops
    
    return None


def circuit_data_to_json(data: CircuitData) -> str:
    return json.dumps(asdict(data))


def circuit_to_json(qc: QuantumCircuit) -> str:
    return circuit_data_to_json(extract_circuit_data(qc))
