"""
metalq/circuit.py - Quantum Circuit Class

Qiskitライクな構文で量子回路を構築。
メソッドチェーンをサポートし、簡潔な記述が可能。

Example:
    # 基本的な使い方
    qc = Circuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    # メソッドチェーン
    qc = Circuit(3).h(0).cx(0, 1).cx(1, 2).measure_all()
    
    # パラメータ化
    theta = Parameter('θ')
    qc = Circuit(1).ry(theta, 0)
    bound_qc = qc.bind_parameters({theta: 0.5})
"""
from __future__ import annotations
from typing import List, Dict, Optional, Union, Any, Set, Tuple
from dataclasses import dataclass, field
import copy

from .parameter import Parameter, ParameterExpression, is_parameterized, get_parameters


@dataclass
class Gate:
    """
    Internal representation of a quantum gate.
    
    Attributes:
        name: Gate name (lowercase)
        qubits: List of qubit indices
        params: List of parameters (float or Parameter)
    """
    name: str
    qubits: List[int]
    params: List[Union[float, Parameter, ParameterExpression]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'qubits': self.qubits,
            'params': [
                p if isinstance(p, (int, float)) else str(p) 
                for p in self.params
            ]
        }


class Circuit:
    """
    Quantum Circuit with Qiskit-like API.
    
    Attributes:
        num_qubits: Number of qubits
        num_clbits: Number of classical bits
        
    Example:
        qc = Circuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.measure_all()
        
        print(qc.draw())
    """
    
    __slots__ = ('num_qubits', 'num_clbits', '_gates', '_measurements', '_parameters')
    
    def __init__(self, num_qubits: int, num_clbits: Optional[int] = None):
        """
        Initialize a quantum circuit.
        
        Args:
            num_qubits: Number of qubits
            num_clbits: Number of classical bits (default: same as num_qubits)
        """
        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive")
        
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits if num_clbits is not None else num_qubits
        self._gates: List[Gate] = []
        self._measurements: List[Tuple[int, int]] = []  # (qubit, clbit)
        self._parameters: Set[Parameter] = set()
    
    # ========================================================================
    # Single-Qubit Gates
    # ========================================================================
    
    def _validate_qubit(self, qubit: int, name: str = "qubit"):
        """Validate qubit index."""
        if not 0 <= qubit < self.num_qubits:
            raise ValueError(f"{name} {qubit} out of range [0, {self.num_qubits})")
    
    def _add_gate(self, name: str, qubits: List[int], 
                  params: Optional[List[Union[float, Parameter]]] = None) -> Circuit:
        """Internal method to add a gate."""
        for q in qubits:
            self._validate_qubit(q)
        
        gate = Gate(name=name, qubits=qubits, params=params or [])
        self._gates.append(gate)
        
        # Track parameters
        for p in (params or []):
            self._parameters.update(get_parameters(p))
        
        return self
    
    def id(self, qubit: int) -> Circuit:
        """Identity gate (no-op, useful for timing)."""
        return self._add_gate('id', [qubit])
    
    def x(self, qubit: int) -> Circuit:
        """Pauli-X (NOT) gate."""
        return self._add_gate('x', [qubit])
    
    def y(self, qubit: int) -> Circuit:
        """Pauli-Y gate."""
        return self._add_gate('y', [qubit])
    
    def z(self, qubit: int) -> Circuit:
        """Pauli-Z gate."""
        return self._add_gate('z', [qubit])
    
    def h(self, qubit: int) -> Circuit:
        """Hadamard gate."""
        return self._add_gate('h', [qubit])
    
    def s(self, qubit: int) -> Circuit:
        """S gate (sqrt(Z), phase gate with θ=π/2)."""
        return self._add_gate('s', [qubit])
    
    def sdg(self, qubit: int) -> Circuit:
        """S-dagger gate (inverse of S)."""
        return self._add_gate('sdg', [qubit])
    
    def t(self, qubit: int) -> Circuit:
        """T gate (sqrt(S), phase gate with θ=π/4)."""
        return self._add_gate('t', [qubit])
    
    def tdg(self, qubit: int) -> Circuit:
        """T-dagger gate (inverse of T)."""
        return self._add_gate('tdg', [qubit])
    
    def sx(self, qubit: int) -> Circuit:
        """Sqrt(X) gate."""
        return self._add_gate('sx', [qubit])
    
    def sxdg(self, qubit: int) -> Circuit:
        """Sqrt(X)-dagger gate."""
        return self._add_gate('sxdg', [qubit])
    
    # ========================================================================
    # Rotation Gates (Parameterizable)
    # ========================================================================
    
    def rx(self, theta: Union[float, Parameter], qubit: int) -> Circuit:
        """Rotation around X-axis by angle theta."""
        return self._add_gate('rx', [qubit], [theta])
    
    def ry(self, theta: Union[float, Parameter], qubit: int) -> Circuit:
        """Rotation around Y-axis by angle theta."""
        return self._add_gate('ry', [qubit], [theta])
    
    def rz(self, theta: Union[float, Parameter], qubit: int) -> Circuit:
        """Rotation around Z-axis by angle theta."""
        return self._add_gate('rz', [qubit], [theta])
    
    def p(self, theta: Union[float, Parameter], qubit: int) -> Circuit:
        """Phase gate (rotation around Z with global phase)."""
        return self._add_gate('p', [qubit], [theta])
    
    def u(self, theta: Union[float, Parameter], 
          phi: Union[float, Parameter], 
          lam: Union[float, Parameter], 
          qubit: int) -> Circuit:
        """
        Universal single-qubit gate U(θ, φ, λ).
        
        U(θ, φ, λ) = [[cos(θ/2), -e^(iλ)sin(θ/2)],
                      [e^(iφ)sin(θ/2), e^(i(φ+λ))cos(θ/2)]]
        """
        return self._add_gate('u', [qubit], [theta, phi, lam])
    
    def u1(self, lam: Union[float, Parameter], qubit: int) -> Circuit:
        """U1 gate (equivalent to p gate)."""
        return self._add_gate('u1', [qubit], [lam])
    
    def u2(self, phi: Union[float, Parameter], 
           lam: Union[float, Parameter], qubit: int) -> Circuit:
        """U2 gate."""
        return self._add_gate('u2', [qubit], [phi, lam])
    
    def u3(self, theta: Union[float, Parameter], 
           phi: Union[float, Parameter], 
           lam: Union[float, Parameter], 
           qubit: int) -> Circuit:
        """U3 gate (same as u)."""
        return self._add_gate('u3', [qubit], [theta, phi, lam])
    
    def r(self, theta: Union[float, Parameter], 
          phi: Union[float, Parameter], qubit: int) -> Circuit:
        """Rotation gate R(θ, φ) = exp(-i θ/2 (cos(φ)X + sin(φ)Y))."""
        return self._add_gate('r', [qubit], [theta, phi])
    
    # ========================================================================
    # Two-Qubit Gates
    # ========================================================================
    
    def cx(self, control: int, target: int) -> Circuit:
        """CNOT (Controlled-X) gate."""
        return self._add_gate('cx', [control, target])
    
    def cnot(self, control: int, target: int) -> Circuit:
        """Alias for cx."""
        return self.cx(control, target)
    
    def cy(self, control: int, target: int) -> Circuit:
        """Controlled-Y gate."""
        return self._add_gate('cy', [control, target])
    
    def cz(self, control: int, target: int) -> Circuit:
        """Controlled-Z gate."""
        return self._add_gate('cz', [control, target])
    
    def ch(self, control: int, target: int) -> Circuit:
        """Controlled-Hadamard gate."""
        return self._add_gate('ch', [control, target])
    
    def cs(self, control: int, target: int) -> Circuit:
        """Controlled-S gate."""
        return self._add_gate('cs', [control, target])
    
    def csdg(self, control: int, target: int) -> Circuit:
        """Controlled-S-dagger gate."""
        return self._add_gate('csdg', [control, target])
    
    def csx(self, control: int, target: int) -> Circuit:
        """Controlled-Sqrt(X) gate."""
        return self._add_gate('csx', [control, target])
    
    def swap(self, qubit1: int, qubit2: int) -> Circuit:
        """SWAP gate."""
        return self._add_gate('swap', [qubit1, qubit2])
    
    def iswap(self, qubit1: int, qubit2: int) -> Circuit:
        """iSWAP gate."""
        return self._add_gate('iswap', [qubit1, qubit2])
    
    def dcx(self, qubit1: int, qubit2: int) -> Circuit:
        """Double CNOT gate (DCX)."""
        return self._add_gate('dcx', [qubit1, qubit2])
    
    def ecr(self, qubit1: int, qubit2: int) -> Circuit:
        """Echoed Cross-Resonance gate."""
        return self._add_gate('ecr', [qubit1, qubit2])
    
    # Controlled rotations
    def crx(self, theta: Union[float, Parameter], control: int, target: int) -> Circuit:
        """Controlled-RX gate."""
        return self._add_gate('crx', [control, target], [theta])
    
    def cry(self, theta: Union[float, Parameter], control: int, target: int) -> Circuit:
        """Controlled-RY gate."""
        return self._add_gate('cry', [control, target], [theta])
    
    def crz(self, theta: Union[float, Parameter], control: int, target: int) -> Circuit:
        """Controlled-RZ gate."""
        return self._add_gate('crz', [control, target], [theta])
    
    def cp(self, theta: Union[float, Parameter], control: int, target: int) -> Circuit:
        """Controlled-Phase gate."""
        return self._add_gate('cp', [control, target], [theta])
    
    def cu(self, theta: Union[float, Parameter], 
           phi: Union[float, Parameter], 
           lam: Union[float, Parameter], 
           gamma: Union[float, Parameter],
           control: int, target: int) -> Circuit:
        """Controlled-U gate with global phase."""
        return self._add_gate('cu', [control, target], [theta, phi, lam, gamma])
    
    def cu1(self, lam: Union[float, Parameter], control: int, target: int) -> Circuit:
        """Controlled-U1 gate."""
        return self._add_gate('cu1', [control, target], [lam])
    
    def cu3(self, theta: Union[float, Parameter], 
            phi: Union[float, Parameter], 
            lam: Union[float, Parameter],
            control: int, target: int) -> Circuit:
        """Controlled-U3 gate."""
        return self._add_gate('cu3', [control, target], [theta, phi, lam])
    
    # Ising coupling gates
    def rxx(self, theta: Union[float, Parameter], qubit1: int, qubit2: int) -> Circuit:
        """RXX (Ising XX coupling) gate."""
        return self._add_gate('rxx', [qubit1, qubit2], [theta])
    
    def ryy(self, theta: Union[float, Parameter], qubit1: int, qubit2: int) -> Circuit:
        """RYY (Ising YY coupling) gate."""
        return self._add_gate('ryy', [qubit1, qubit2], [theta])
    
    def rzz(self, theta: Union[float, Parameter], qubit1: int, qubit2: int) -> Circuit:
        """RZZ (Ising ZZ coupling) gate."""
        return self._add_gate('rzz', [qubit1, qubit2], [theta])
    
    def rzx(self, theta: Union[float, Parameter], qubit1: int, qubit2: int) -> Circuit:
        """RZX gate."""
        return self._add_gate('rzx', [qubit1, qubit2], [theta])
    
    # ========================================================================
    # Three-Qubit Gates
    # ========================================================================
    
    def ccx(self, control1: int, control2: int, target: int) -> Circuit:
        """Toffoli (CCX) gate."""
        return self._add_gate('ccx', [control1, control2, target])
    
    def toffoli(self, control1: int, control2: int, target: int) -> Circuit:
        """Alias for ccx."""
        return self.ccx(control1, control2, target)
    
    def cswap(self, control: int, target1: int, target2: int) -> Circuit:
        """Fredkin (CSWAP) gate."""
        return self._add_gate('cswap', [control, target1, target2])
    
    def fredkin(self, control: int, target1: int, target2: int) -> Circuit:
        """Alias for cswap."""
        return self.cswap(control, target1, target2)
    
    def ccz(self, qubit1: int, qubit2: int, qubit3: int) -> Circuit:
        """CCZ gate."""
        return self._add_gate('ccz', [qubit1, qubit2, qubit3])
    
    # ========================================================================
    # Measurement
    # ========================================================================
    
    def measure(self, qubit: int, clbit: int) -> Circuit:
        """
        Measure a qubit into a classical bit.
        
        Args:
            qubit: Qubit index to measure
            clbit: Classical bit to store result
        """
        self._validate_qubit(qubit)
        if not 0 <= clbit < self.num_clbits:
            raise ValueError(f"clbit {clbit} out of range [0, {self.num_clbits})")
        
        self._measurements.append((qubit, clbit))
        return self
    
    def measure_all(self) -> Circuit:
        """Measure all qubits into corresponding classical bits."""
        for i in range(min(self.num_qubits, self.num_clbits)):
            self._measurements.append((i, i))
        return self
    
    # ========================================================================
    # Barrier (for visualization/optimization hints)
    # ========================================================================
    
    def barrier(self, *qubits: int) -> Circuit:
        """
        Add a barrier (visual separator, optimization boundary).
        
        Args:
            *qubits: Qubits to include in barrier. If empty, all qubits.
        """
        if not qubits:
            qubits = tuple(range(self.num_qubits))
        return self._add_gate('barrier', list(qubits))
    
    # ========================================================================
    # Parameter Binding
    # ========================================================================
    
    @property
    def parameters(self) -> List[Parameter]:
        """Get list of unbound parameters (in order of first appearance)."""
        seen = set()
        ordered = []
        for gate in self._gates:
            for p in gate.params:
                for param in get_parameters(p):
                    if param not in seen:
                        seen.add(param)
                        ordered.append(param)
        return ordered
    
    @property
    def num_parameters(self) -> int:
        """Get number of unique parameters."""
        return len(self._parameters)
    
    def bind_parameters(self, 
                        params: Union[Dict[Parameter, float], List[float]]) -> Circuit:
        """
        Bind parameter values to create a concrete circuit.
        
        Args:
            params: Dict mapping Parameters to values, or list of values
                    (in order of self.parameters)
        
        Returns:
            New circuit with bound parameters
            
        Example:
            theta = Parameter('θ')
            qc = Circuit(1).ry(theta, 0)
            
            # Dict binding
            bound = qc.bind_parameters({theta: 0.5})
            
            # List binding
            bound = qc.bind_parameters([0.5])
        """
        from .parameter import evaluate_parameter
        
        # Convert list to dict
        if isinstance(params, (list, tuple)):
            param_list = self.parameters
            if len(params) != len(param_list):
                raise ValueError(
                    f"Expected {len(param_list)} parameters, got {len(params)}"
                )
            params = dict(zip(param_list, params))
        
        # Create new circuit
        new_circuit = Circuit(self.num_qubits, self.num_clbits)
        new_circuit._measurements = self._measurements.copy()
        
        # Bind each gate's parameters
        for gate in self._gates:
            new_params = []
            for p in gate.params:
                if is_parameterized(p):
                    new_params.append(evaluate_parameter(p, params))
                else:
                    new_params.append(p)
            
            new_circuit._gates.append(Gate(
                name=gate.name,
                qubits=gate.qubits.copy(),
                params=new_params
            ))
        
        return new_circuit
    
    # ========================================================================
    # Properties
    # ========================================================================
    
    @property
    def depth(self) -> int:
        """
        Calculate circuit depth (critical path length).
        
        Returns:
            Number of layers in the circuit
        """
        if not self._gates:
            return 0
        
        qubit_depths = [0] * self.num_qubits
        
        for gate in self._gates:
            if gate.name == 'barrier':
                continue
            
            # Gate depth is max of all involved qubits + 1
            max_depth = max(qubit_depths[q] for q in gate.qubits)
            new_depth = max_depth + 1
            
            for q in gate.qubits:
                qubit_depths[q] = new_depth
        
        return max(qubit_depths) if qubit_depths else 0
    
    @property
    def size(self) -> int:
        """Get total number of gates (excluding barriers)."""
        return sum(1 for g in self._gates if g.name != 'barrier')
    
    @property
    def gates(self) -> List[Gate]:
        """Get list of gates (read-only copy)."""
        return self._gates.copy()
    
    @property
    def measurements(self) -> List[Tuple[int, int]]:
        """Get list of measurements (qubit, clbit) pairs."""
        return self._measurements.copy()
    
    def has_measurements(self) -> bool:
        """Check if circuit has any measurements."""
        return len(self._measurements) > 0
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def copy(self) -> Circuit:
        """Create a deep copy of this circuit."""
        new_circuit = Circuit(self.num_qubits, self.num_clbits)
        new_circuit._gates = [
            Gate(g.name, g.qubits.copy(), g.params.copy()) 
            for g in self._gates
        ]
        new_circuit._measurements = self._measurements.copy()
        new_circuit._parameters = self._parameters.copy()
        return new_circuit
    
    def inverse(self) -> Circuit:
        """
        Return the inverse (adjoint) of this circuit.
        
        Returns:
            New circuit with gates in reverse order and inverted
        """
        new_circuit = Circuit(self.num_qubits, self.num_clbits)
        
        for gate in reversed(self._gates):
            inv_gate = self._invert_gate(gate)
            new_circuit._gates.append(inv_gate)
        
        new_circuit._parameters = self._parameters.copy()
        return new_circuit
    
    def _invert_gate(self, gate: Gate) -> Gate:
        """Get the inverse of a gate."""
        name = gate.name
        
        # Self-inverse gates
        if name in {'x', 'y', 'z', 'h', 'cx', 'cy', 'cz', 'swap', 'ccx', 'ccz', 'cswap', 'barrier'}:
            return Gate(name, gate.qubits.copy(), gate.params.copy())
        
        # S <-> Sdg, T <-> Tdg
        inverse_map = {
            's': 'sdg', 'sdg': 's',
            't': 'tdg', 'tdg': 't',
            'sx': 'sxdg', 'sxdg': 'sx',
        }
        if name in inverse_map:
            return Gate(inverse_map[name], gate.qubits.copy(), [])
        
        # Rotation gates: negate angle
        if name in {'rx', 'ry', 'rz', 'p', 'u1', 'crx', 'cry', 'crz', 'cp', 'rxx', 'ryy', 'rzz', 'rzx'}:
            new_params = [-p for p in gate.params]
            return Gate(name, gate.qubits.copy(), new_params)
        
        # U gate: more complex inversion
        if name in {'u', 'u3', 'cu', 'cu3'}:
            # U†(θ,φ,λ) = U(-θ, -λ, -φ)
            theta, phi, lam = gate.params[:3]
            new_params = [-theta, -lam, -phi] + gate.params[3:]
            return Gate(name, gate.qubits.copy(), new_params)
        
        # Default: return as-is (may not be correct for all gates)
        return Gate(name, gate.qubits.copy(), gate.params.copy())
    
    def compose(self, other: Circuit, inplace: bool = False) -> Circuit:
        """
        Compose with another circuit (append other's gates).
        
        Args:
            other: Circuit to append
            inplace: If True, modify this circuit
            
        Returns:
            Composed circuit
        """
        if other.num_qubits != self.num_qubits:
            raise ValueError(
                f"Cannot compose circuits with different qubit counts: "
                f"{self.num_qubits} vs {other.num_qubits}"
            )
        
        if inplace:
            result = self
        else:
            result = self.copy()
        
        result._gates.extend(g for g in other._gates)
        result._measurements.extend(other._measurements)
        result._parameters.update(other._parameters)
        
        return result
    
    def __add__(self, other: Circuit) -> Circuit:
        """Compose circuits using + operator."""
        return self.compose(other)
    
    def __iadd__(self, other: Circuit) -> Circuit:
        """In-place compose using += operator."""
        return self.compose(other, inplace=True)
    
    # ========================================================================
    # Serialization
    # ========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert circuit to dictionary for serialization.
        
        Used for passing to native backend.
        """
        return {
            'num_qubits': self.num_qubits,
            'num_clbits': self.num_clbits,
            'gates': [g.to_dict() for g in self._gates],
            'measurements': self._measurements,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Circuit:
        """Create circuit from dictionary."""
        circuit = cls(data['num_qubits'], data.get('num_clbits'))
        
        for g in data['gates']:
            circuit._gates.append(Gate(
                name=g['name'],
                qubits=g['qubits'],
                params=g.get('params', [])
            ))
        
        circuit._measurements = [tuple(m) for m in data.get('measurements', [])]
        return circuit
    
    # ========================================================================
    # String Representation
    # ========================================================================
    
    def __repr__(self) -> str:
        return (f"Circuit(qubits={self.num_qubits}, "
                f"gates={self.size}, depth={self.depth})")
    
    def __str__(self) -> str:
        return self.draw()
    
    def draw(self, output: str = 'text') -> str:
        """
        Draw the circuit.
        
        Args:
            output: 'text' for ASCII art, 'latex' for LaTeX source
        
        Returns:
            String representation of the circuit
        """
        from .visualization import draw_circuit
        return draw_circuit(self, output=output)
