"""
metalq/visualization.py - Circuit Visualization

Supports ASCII/Unicode art drawing of quantum circuits with rounded corners.
"""
from typing import List, Dict, Any, Tuple, Optional, Set
import math

if False:  # TYPE_CHECKING
    from .circuit import Circuit, Gate

# Box Drawing Components (Rounded)
H_LINE = '─'
V_LINE = '│'
CROSS  = '┼'
R_TOP_L = '╭'
R_TOP_R = '╮'
R_BOT_L = '╰'
R_BOT_R = '╯'
T_DOWN = '┬'  # For control top
T_UP   = '┴'  # For control bottom
L_T    = '┤'
R_T    = '├'
CTRL   = '●'
MEAS   = 'M' # Or generic symbol

class TextDrawer:
    def __init__(self, circuit: 'Circuit'):
        self.circuit = circuit
        self.n_qubits = circuit.num_qubits
        
        # Grid management
        # (x, y) -> char
        # y coordinates: qubit i is at y = i * 3 + 1
        self.grid: Dict[Tuple[int, int], str] = {}
        
        # Track current x logical position for each qubit
        # This determines where the next gate can be placed
        self.qubit_x = [0] * self.n_qubits
        
        # Real column widths (logic x -> character width)
        self.col_widths: Dict[int, int] = {}
        
        # Mapping from logic x to accumulated character x is done at rendering
        
    def draw(self) -> str:
        # 1. Draw Labels
        self._draw_labels()
        
        # 2. Process Gates
        for gate in self.circuit.gates:
            self._place_gate(gate)
            
        # 3. Process Measurements
        #   (Treated as gates usually, but if stored separately in circuit...)
        if hasattr(self.circuit, 'measurements') and self.circuit.measurements:
            for q, c in self.circuit.measurements:
                self._place_measurement(q, c)
                
        # 4. Render to string
        return self._render()
        
    def _draw_labels(self):
        """Draw initial labels (q_0, etc.) at x=-1 (logical)."""
        # This is handled during rendering as a prefix, 
        # but we can set initial x to 0.
        pass

    def _place_gate(self, gate):
        name = gate.name.upper()
        qubits = gate.qubits
        params = gate.params
        
        # Label generation
        label = name
        if params:
            # Simple parameter formatting
            p_strs = []
            for p in params:
                 if isinstance(p, float):
                     p_strs.append(f"{p:.2f}")
                 else:
                     p_strs.append(str(p))
            if p_strs:
                label += f"({','.join(p_strs)})"
                
        # Determine gate type
        is_control = name in ('CX', 'CNOT', 'CZ', 'CRX', 'CRY', 'CRZ', 'CP', 'CU', 'CSWAP')
        is_swap = name == 'SWAP'
        
        # Determine logical X position
        # Must be after current x of all involved qubits (including those spanned by vertical lines)
        min_q = min(qubits)
        max_q = max(qubits)
        width_range = range(min_q, max_q + 1)
        
        current_max_x = max(self.qubit_x[q] for q in width_range)
        place_x = current_max_x
        
        # Register the gate width
        # Box width = label length + 2 (padding)
        # Control width = 1? But to align with box, maybe same width or center?
        # We'll use variable width columns.
        box_width = len(label) + 2
        
        if is_control:
            # For CNOT, target is box (X or generic), control is dot
            # If name is big (CRX), box is big. Dot should be centered.
            # We treat the whole column as having `box_width` width.
            
            # Identify controls and target
            # Heuristic: verify against known gates or just treat first N-1 as controls?
            # Implementation specific. 
            # In metalq/circuit.py, cx(ctrl, tgt). controls=[ctrl], target is last?
            # Standard gates: last one is target usually, except SWAP/CSWAP.
            
            # Let's simplify: 
            # If gate name starts with 'C', assume last qubit is target, others controls?
            # Works for CX, CY, CZ, CRX...
            # CSWAP: last 2 are targets?
            # Let's just draw generic boxes for everything except standard CX.
            
            if name in ('CX', 'CNOT'):
                target = qubits[-1]
                controls = qubits[:-1]
                self._draw_cnot(place_x, controls, target)
                self.col_widths[place_x] = max(self.col_widths.get(place_x, 0), 3)
            elif name == 'CZ':
                # All dots
                self._draw_cz(place_x, qubits)
                self.col_widths[place_x] = max(self.col_widths.get(place_x, 0), 3)
            elif name == 'SWAP':
                 self._draw_swap(place_x, qubits)
                 self.col_widths[place_x] = max(self.col_widths.get(place_x, 0), 3)
            else:
                 # Generic controlled gate
                 # Box on target, dots on controls
                 target = qubits[-1]
                 controls = qubits[:-1]
                 self._draw_controlled_box(place_x, controls, target, label)
                 self.col_widths[place_x] = max(self.col_widths.get(place_x, 0), box_width)
        
        elif name == 'BARRIER':
            # Draw barrier
            for q in qubits:
                 self._set_grid(place_x, q * 3 + 1, '░')
            self.col_widths[place_x] = max(self.col_widths.get(place_x, 0), 1)
            
        else:
            # Standard Box Gate
            self._draw_box(place_x, qubits, label)
            self.col_widths[place_x] = max(self.col_widths.get(place_x, 0), box_width)
            
        # Update cursor x
        for q in width_range:
            self.qubit_x[q] = place_x + 1
            
    def _place_measurement(self, qubit, clbit):
        x = self.qubit_x[qubit]
        label = f"M({clbit})"
        
        # Draw box
        self._draw_box(x, [qubit], label)
        self.col_widths[x] = max(self.col_widths.get(x, 0), len(label) + 2)
        
        self.qubit_x[qubit] = x + 1

    def _draw_box(self, x, qubits, label):
        """Draw a box covering specified qubits."""
        min_q = min(qubits)
        max_q = max(qubits)
        
        top_y = min_q * 3
        bot_y = max_q * 3 + 2
        
        # Box content is centered string
        # We store just the label in the grid? 
        # No, we need to store individual chars because we render to fixed grid later?
        # Better: store metadata objects in grid, render later.
        # But to be simple, let's store special rendering markers.
        
        # Marker: ('BOX', label, start_q, end_q)
        # But we need to handle overlaps.
        # Let's assume layout is column-based.
        # We'll just define the layout instructions per column.
        pass # Logic moved to render 
        
        # To avoid complex grid logic, let's store "Drawing Instructions" per (x, y)
        # Actually doing it directly on sparse grid (x, y, char) is fine if we resolve coordinates from col_widths.
        
        # Wait, if we write to self.grid with LOGICAL X, we can't map to chars easily because widths vary.
        # We need to map Logical X -> Real X.
        # But we don't know Real X until we know all column widths.
        # So: First pass calculates widths (done in _place_gate).
        # Second pass (render) draws.
        
        # Store "Action" at logical X
        if x not in self.actions: self.actions[x] = []
        self.actions[x].append(('BOX', qubits, label))

    # Redesign: Two pass approach.
    # Pass 1: _place_gate just determines widths and dependencies. Store 'Ops' in a list.
    # Pass 2: Render.
    
    # ... Wait, I can't rewrite __init__ easily if I'm inside methods.
    # I'll restart the class definition structure.
    
class TextDrawer:
    def __init__(self, circuit: 'Circuit'):
        self.circuit = circuit
        self.n_qubits = circuit.num_qubits
        self.col_widths: Dict[int, int] = {}
        self.qubit_x = [0] * self.n_qubits
        self.ops = {} # x -> list of ops

    def draw(self) -> str:
        # Pass 1: Layout
        for gate in self.circuit.gates:
            self._layout_gate(gate)
            
        for q, c in self.circuit.measurements:
            self._layout_measurement(q, c)
            
        # Pass 2: Render
        return self._render()
        
    def _layout_gate(self, gate):
        name = gate.name.upper()
        qubits = gate.qubits
        params = gate.params
        
        label = name
        if params:
            p_vals = [f"{p:.2f}" if isinstance(p, float) else str(p) for p in params]
            label += f"({','.join(p_vals)})"
            
        width_range = range(min(qubits), max(qubits) + 1)
        x = max(self.qubit_x[q] for q in width_range)
        
        box_width = len(label) + 2
        
        # Determine op type
        if name in ('CX', 'CNOT'):
            op = ('CNOT', qubits[:-1], qubits[-1])
            self.add_op(x, op, 3)
        elif name == 'CZ':
            op = ('CZ', qubits)
            self.add_op(x, op, 3)
        elif name == 'SWAP':
            op = ('SWAP', qubits)
            self.add_op(x, op, 3)
        elif name == 'BARRIER':
            op = ('BARRIER', qubits)
            self.add_op(x, op, 1)
        # elif name.startswith('C') and len(qubits) > 1: # Generic controlled
            # Assume last is target
            # op = ('CBOX', qubits[:-1], qubits[-1], label)
            # self.add_op(x, op, box_width)
        else:
            op = ('BOX', qubits, label)
            self.add_op(x, op, box_width)
            
        for q in width_range:
            self.qubit_x[q] = x + 1
            
    def _layout_measurement(self, qubit, clbit):
        x = self.qubit_x[qubit]
        label = f"M({clbit})"
        self.add_op(x, ('BOX', [qubit], label), len(label) + 2)
        self.qubit_x[qubit] = x + 1
        
    def add_op(self, x, op, width):
        if x not in self.ops: self.ops[x] = []
        self.ops[x].append(op)
        w = self.col_widths.get(x, 0)
        self.col_widths[x] = max(w, width)
        
    def _render(self) -> str:
        if not self.ops:
            max_x = 0
        else:
            max_x = max(self.ops.keys()) + 1
        
        # Labels
        labels = [f"q_{i}: " for i in range(self.n_qubits)]
        label_width = max(len(l) for l in labels)
        
        # Initialize grid rows
        # Each qubit has 3 rows: top(0), mid(1), bot(2)
        # Total rows = n_qubits * 3
        # wire is at local row 1
        grid = ["" for _ in range(self.n_qubits * 3)]
        
        # Add labels
        for i in range(self.n_qubits):
            # Mid row gets label
            grid[i*3 + 1] = labels[i].rjust(label_width)
            # Others empty
            grid[i*3]     = " " * label_width
            grid[i*3 + 2] = " " * label_width
            
        # Iterate columns
        for x in range(max_x):
            ops = self.ops.get(x, [])
            width = self.col_widths.get(x, 3) # default min width 3
            if width % 2 == 0: width += 1 # Ensure odd width for centering
            
            # Prepare column buffer
            # Initialize with wires
            col_buf = [[" "]*width for _ in range(self.n_qubits * 3)]
            for i in range(self.n_qubits):
                 # Wire at mid
                 for c in range(width): col_buf[i*3+1][c] = H_LINE
            
            # Draw ops
            for op in ops:
                type = op[0]
                center_x = width // 2
                
                if type == 'BOX':
                    qubits, label = op[1], op[2]
                    self._draw_box_in_col(col_buf, qubits, label, width)
                
                elif type == 'CNOT':
                    ctrls, tgt = op[1], op[2]
                    self._draw_cnot_in_col(col_buf, ctrls, tgt, width)
                    
                elif type == 'CZ':
                    qubits = op[1]
                    self._draw_cz_in_col(col_buf, qubits, width)
                    
                elif type == 'SWAP':
                    qubits = op[1]
                    self._draw_swap_in_col(col_buf, qubits, width)
                
                elif type == 'BARRIER':
                    qubits = op[1]
                    for q in qubits:
                        col_buf[q*3+1][center_x] = '░'

            # Append to grid
            for r in range(len(grid)):
                grid[r] += "".join(col_buf[r])
                
        # Trim empty rows? No, keep layout
        # But maybe remove spacer rows (rows 2, 5, etc if they are just spaces?)
        # q0 bot is row 2. q1 top is row 3.
        # If no vertical lines, they are empty.
        
        return "\n".join(grid)

    def _draw_box_in_col(self, buf, qubits, label, width):
        min_q, max_q = min(qubits), max(qubits)
        
        # Draw vertical connections if multi-qubit
        center_x = width // 2
        if min_q != max_q:
             # Draw vertical line through everything in between
             start_y = min_q * 3 + 1
             end_y   = max_q * 3 + 1
             for y in range(start_y, end_y + 1):
                 # If we overwrite a wire H_LINE, make it CROSS?
                 # Inside the box is different.
                 # Box covers everything.
                 pass
        
        # For rounded box:
        # Top boundary at min_q top (row min_q*3)
        # Bottom boundary at max_q bot (row max_q*3 + 2)
        top_y = min_q * 3
        bot_y = max_q * 3 + 2
        
        label_start = center_x - len(label)//2
        
        # Top Line
        buf[top_y][center_x - len(label)//2 - 1] = R_TOP_L
        for i in range(len(label)): buf[top_y][label_start + i] = H_LINE
        buf[top_y][label_start + len(label)] = R_TOP_R
        
        # Bottom Line
        buf[bot_y][center_x - len(label)//2 - 1] = R_BOT_L
        for i in range(len(label)): buf[bot_y][label_start + i] = H_LINE
        buf[bot_y][label_start + len(label)] = R_BOT_R
        
        # Sides and content
        # Middle rows
        for y in range(top_y + 1, bot_y):
            # Left/Right walls
            buf[y][label_start - 1] = V_LINE
            buf[y][label_start + len(label)] = V_LINE
            
            # If this is a wire (y % 3 == 1), replace wire with label part or space
            if y % 3 == 1:
                q_idx = y // 3
                # Check if this qubit is part of the gate?
                # Usually box covers all.
                # If qubit is in 'qubits', display label on the middle qubit?
                # Or repeat label on every qubit line?
                # Standard: Center label in the box vertically.
                pass
            
            # Clear inside
            for k in range(len(label)):
                buf[y][label_start + k] = " "
                
        # Place label text at vertical center
        mid_y = (top_y + bot_y) // 2
        for k, char in enumerate(label):
            buf[mid_y][label_start + k] = char

    def _draw_cnot_in_col(self, buf, ctrls, tgt, width):
        center_x = width // 2
        all_q = ctrls + [tgt]
        min_q, max_q = min(all_q), max(all_q)
        
        # Vertical Line
        for q in range(min_q, max_q + 1):
            y_wire = q * 3 + 1
            # Top/Bot of this qubit
            # Draw line through
            for y in [q*3, q*3+1, q*3+2]:
                 if min_q*3+1 <= y <= max_q*3+1:
                     # Crossing wire?
                     if y == y_wire:
                         buf[y][center_x] = CROSS
                     else:
                         buf[y][center_x] = V_LINE
                         
        # Controls
        for q in ctrls:
            buf[q*3+1][center_x] = CTRL
            
        # Target
        # Circle plus? Or just X in text?
        # Unicode ⊕ U+2295? Or (+)
        # Let's use (+).
        buf[tgt*3+1][center_x] = 'X' # Or ⊕ if supported font
        # Overwrite cross
        
    def _draw_cz_in_col(self, buf, qubits, width):
        center_x = width // 2
        min_q, max_q = min(qubits), max(qubits)
        
        # Vertical Line
        for q in range(min_q, max_q + 1):
            y_wire = q * 3 + 1
            for y in [q*3, q*3+1, q*3+2]:
                 if min_q*3+1 <= y <= max_q*3+1:
                     if y == y_wire:
                         buf[y][center_x] = CROSS
                     else:
                         buf[y][center_x] = V_LINE
        
        for q in qubits:
            buf[q*3+1][center_x] = CTRL

    def _draw_swap_in_col(self, buf, qubits, width):
        center_x = width // 2
        min_q, max_q = min(qubits), max(qubits)
        
        for q in range(min_q, max_q + 1):
             y_wire = q * 3 + 1
             for y in [q*3, q*3+1, q*3+2]:
                 if min_q*3+1 <= y <= max_q*3+1:
                     if y == y_wire:
                         buf[y][center_x] = CROSS
                     else:
                         buf[y][center_x] = V_LINE
                         
        for q in qubits:
            buf[q*3+1][center_x] = 'x'

def draw_circuit(circuit: 'Circuit', output: str = 'text') -> str:
    if output == 'text':
        drawer = TextDrawer(circuit)
        return drawer.draw()
    else:
        return "LaTeX not implemented"

