"""
metalq/visualization.py - Circuit Visualization

Supports ASCII art drawing of quantum circuits.
"""
from typing import List, Dict, Any, Tuple, Optional
import math

if False:  # TYPE_CHECKING with circular import workaround
    from .circuit import Circuit, Gate


def draw_circuit(circuit: 'Circuit', output: str = 'text') -> str:
    """
    Draw a quantum circuit.
    
    Args:
        circuit: Circuit to draw
        output: 'text' or 'latex' (latex not fully implemented yet)
        
    Returns:
        String representation
    """
    if output == 'text':
        return _draw_text(circuit)
    elif output == 'latex':
        return _draw_latex(circuit)
    else:
        raise ValueError(f"Unknown output format: {output}")


def _draw_text(circuit: 'Circuit') -> str:
    """Draw circuit using ASCII art."""
    n_qubits = circuit.num_qubits
    n_clbits = circuit.num_clbits
    
    # Grid: 2D array of chars. Each qubit is a row.
    # We build it column by column (gate by gate)
    
    # Initial labels
    qubit_lines = [f"q_{i}: ─" for i in range(n_qubits)]
    
    # Process each gate
    for gate in circuit.gates:
        name = gate.name.upper()
        qubits = gate.qubits
        params = gate.params
        
        # Determine width of the gate block
        param_str = ""
        if params:
            param_vals = [f"{p:.2f}" if isinstance(p, float) else str(p) for p in params]
            param_str = f"({','.join(param_vals)})"
        
        label = name + param_str
        width = len(label) + 2
        
        # Check for control/target structure
        if name in ('CX', 'CNOT'):
            # Special handling for CNOT
            ctrl, tgt = qubits
            _add_multicolumn_gate(qubit_lines, [ctrl], [tgt], "●", "+", width=3)
        
        elif name in ('CZ',):
             ctrl, tgt = qubits
             _add_multicolumn_gate(qubit_lines, [ctrl, tgt], [], "●", "●", width=3)

        elif name in ('SWAP',):
             q1, q2 = qubits
             _add_multicolumn_gate(qubit_lines, [q1, q2], [], "x", "x", width=3)
             
        elif name == 'BARRIER':
             _add_barrier(qubit_lines, qubits)
             
        elif len(qubits) == 1:
            # Single qubit gate
            q = qubits[0]
            _add_single_gate(qubit_lines, q, label)
            
        else:
            # Generic multi-qubit gate
            _add_box_gate(qubit_lines, qubits, label)

    # Measurements
    for q, c in circuit.measurements:
        _add_measurement(qubit_lines, q, c)

    return "\n".join(qubit_lines)


def _pad_lines(lines: List[str]):
    """Pad all lines to the same length."""
    max_len = max(len(line) for line in lines)
    for i in range(len(lines)):
        lines[i] = lines[i].ljust(max_len, '─')


def _add_single_gate(lines: List[str], qubit: int, label: str):
    """Add a single qubit gate."""
    # Pad others
    _pad_lines(lines)
    
    lines[qubit] += f"┤ {label} ├─"


def _add_multicolumn_gate(lines: List[str], controls: List[int], targets: List[int], 
                          ctrl_sym: str, tgt_sym: str, width: int = 3):
    """Add multi-column gate like CNOT."""
    _pad_lines(lines)
    
    all_indices = controls + targets
    min_q = min(all_indices)
    max_q = max(all_indices)
    
    # Center position
    center = width // 2
    
    for i in range(len(lines)):
        if i in controls:
            lines[i] += "".ljust(center, '─') + ctrl_sym + "".ljust(width - center - 1, '─') + "─"
        elif i in targets:
             lines[i] += "".ljust(center, '─') + tgt_sym + "".ljust(width - center - 1, '─') + "─"
        elif min_q < i < max_q:
            # Vertical line crossing
            lines[i] += "".ljust(center, '─') + "┼" + "".ljust(width - center - 1, '─') + "─"
        else:
             lines[i] += "─" * (width + 1)
             
    # TODO: Add vertical lines (connecting controls/targets)
    # This naive implementation relies on the viewer connecting the dots mentally or using a fixed width font where | aligns.
    # Better implementation would draw vertical lines in subsequent steps or using a 2D line buffer.
    # For now, simplistic approach.


def _add_box_gate(lines: List[str], qubits: List[int], label: str):
    """Add a box spanning multiple qubits."""
    _pad_lines(lines)
    
    min_q = min(qubits)
    max_q = max(qubits)
    width = len(label) + 2
    
    for i in range(len(lines)):
        if min_q <= i <= max_q:
            if i == min_q: # top
                if i == max_q: # single line box? shouldn't happen here usually
                     lines[i] += f"┤ {label} ├─"
                else:
                     lines[i] += f"┌─{label}─┐─"
            elif i == max_q: # bottom
                lines[i] += f"└{'─' * (width-2)}┘─"
            else: # middle
                lines[i] += f"│{' ' * (width-2)}│─"
        else:
            lines[i] += "─" * width + "──"


def _add_barrier(lines: List[str], qubits: List[int]):
    """Add barrier."""
    _pad_lines(lines)
    for q in qubits:
        lines[q] += "░─"


def _add_measurement(lines: List[str], qubit: int, clbit: int):
    """Add measurement symbol."""
    _pad_lines(lines)
    lines[qubit] += f"M({clbit})─"


def _draw_latex(circuit: 'Circuit') -> str:
    """Stub for LaTeX drawing."""
    # Simple stub that returns a quantikz environment
    code = "\\begin{quantikz}\n"
    for i in range(circuit.num_qubits):
        code += f"\\lstick{{q_{{{i}}}}} & \\qw & \\qw \\\\\n"
    code += "\\end{quantikz}"
    return code
