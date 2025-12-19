"""
circuit_data.py - High-level gate support and decomposition
"""
import json
import warnings
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction, Measure, Barrier, Reset
from qiskit.circuit.library import UnitaryGate
import numpy as np


# Metal-Q Supported Native Gates
NATIVE_GATES = {
    # 1Q gates
    'id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'sx', 'sxdg',
    'rx', 'ry', 'rz', 'p', 'u', 'u1', 'u2', 'u3', 'r',
    # 2Q gates
    'cx', 'cy', 'cz', 'ch', 'cs', 'csdg', 'csx', 'swap', 'iswap',
    'cp', 'crx', 'cry', 'crz', 'cu', 'cu1', 'cu3',
    'rxx', 'ryy', 'rzz', 'rzx', 'dcx', 'ecr',
    # 3Q gates
    'ccx', 'ccz', 'cswap',
    # Special
    'reset', 'unitary'
}

# Basis gates for decomposition target
BASIS_GATES = [
    'id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'sx',
    'rx', 'ry', 'rz', 'p', 'u',
    'cx', 'cz', 'cy', 'swap', 'cp', 'crz',
    'ccx', 'cswap'
]


@dataclass
class GateData:
    """Data for a single gate sent to Metal backend"""
    name: str
    qubits: List[int]
    params: List[float] = field(default_factory=list)
    matrix: Optional[List[List[List[float]]]] = None


@dataclass  
class CircuitData:
    """Full circuit data payload"""
    num_qubits: int
    num_clbits: int
    gates: List[GateData]
    measurements: List[Tuple[int, int]]


def decompose_to_native(qc: QuantumCircuit, max_reps: int = 15) -> QuantumCircuit:
    """
    Decompose high-level gates into native Metal-Q gates.
    
    Args:
        qc: Input QuantumCircuit
        max_reps: Maximum decomposition depth
    
    Returns:
        Decomposed QuantumCircuit
    """
    decomposed = qc.copy()
    
    # Step 1: Iterative decomposition
    for _ in range(max_reps):
        needs_more = False
        for inst_data in decomposed.data:
            inst = inst_data.operation
            name = inst.name.lower()
            if name not in NATIVE_GATES and not isinstance(inst, (Measure, Barrier, Reset)):
                needs_more = True
                break
        
        if not needs_more:
            break
        
        decomposed = decomposed.decompose()
    
    # Step 2: Transpile remaining unknown gates
    unknown_gates = []
    for inst_data in decomposed.data:
        inst = inst_data.operation
        name = inst.name.lower()
        if name not in NATIVE_GATES and not isinstance(inst, (Measure, Barrier, Reset)):
            unknown_gates.append(name)
    
    if unknown_gates:
        try:
            decomposed = transpile(
                decomposed,
                basis_gates=BASIS_GATES,
                optimization_level=1
            )
        except Exception as e:
            warnings.warn(f"Transpile fallback failed for gates {unknown_gates}: {e}")
    
    return decomposed


def extract_circuit_data(qc: QuantumCircuit, auto_decompose: bool = True) -> CircuitData:
    """
    Convert Qiskit QuantumCircuit to Metal-Q CircuitData format.
    """
    if auto_decompose:
        qc = decompose_to_native(qc)
    
    gates = []
    measurements = []
    
    for instruction_data in qc.data:
        instruction = instruction_data.operation
        qargs = instruction_data.qubits
        cargs = instruction_data.clbits
        
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
    """Convert a single Qiskit Instruction to GateData."""
    name = inst.name.lower()
    
    # Convert params to float
    params = []
    for p in inst.params:
        try:
            val = float(p)
        except (TypeError, ValueError):
            try:
                # Handle complex params if any (take real part or magnitude? usually real for rot angles)
                val = float(complex(p).real)
            except:
                val = 0.0
        params.append(val)
    
    # Handle Reset
    if isinstance(inst, Reset) or name == 'reset':
        return GateData(name='reset', qubits=qubits, params=[])
    
    # Custom Unitary: extract matrix
    if name == 'unitary':
        try:
            matrix = inst.to_matrix()
            matrix_data = _matrix_to_json(matrix)
            return GateData(
                name='unitary',
                qubits=qubits,
                params=[],
                matrix=matrix_data
            )
        except Exception as e:
            warnings.warn(f"Metal-Q: Failed to extract matrix for unitary: {e}")
            return None

    # Native gates
    if name in NATIVE_GATES:
        return GateData(name=name, qubits=qubits, params=params)
    try:
        # qiskit 2.0+ uses standard numpy array return
        matrix = inst.to_matrix()
        matrix_data = _matrix_to_json(matrix)
        return GateData(
            name='unitary',
            qubits=qubits,
            params=[],
            matrix=matrix_data
        )
    except Exception as e:
        warnings.warn(f"Metal-Q: Unsupported gate '{name}' skipped. Error: {e}")
        return None


def _matrix_to_json(matrix: np.ndarray) -> List[List[List[float]]]:
    """Convert numpy matrix to JSON-compatible list of lists (row-major)."""
    matrix_data = []
    for row in matrix:
        row_data = []
        for val in row:
            # Explicitly cast to float for JSON serialization
            # Store as [real, imag] pair
            row_data.append([float(np.float64(val.real)), float(np.float64(val.imag))])
        matrix_data.append(row_data)
    return matrix_data


def circuit_data_to_json(data: CircuitData) -> str:
    """Serialize CircuitData to JSON string."""
    json_str = json.dumps(asdict(data), ensure_ascii=False)
    # print(f"DEBUG JSON: {json_str[:200]}...") 
    return json_str


def circuit_to_json(qc: QuantumCircuit, auto_decompose: bool = True) -> str:
    """Convert QuantumCircuit directly to JSON string."""
    return circuit_data_to_json(extract_circuit_data(qc, auto_decompose))

