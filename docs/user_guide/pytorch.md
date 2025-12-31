# PyTorch Integration

Metal-Q provides seamless integration with PyTorch via `metalq.torch`.

## Quantitative Layer

You can wrap a Metal-Q circuit into a PyTorch `nn.Module` using `QuantumLayer`.

```python
from metalq.torch import QuantumLayer
from metalq import Circuit, Parameter, Hamiltonian, Z

# Define circuit with parameters
qc = Circuit(1)
theta = Parameter('theta')
qc.rx(theta, 0)

# Define observable
H = Z(0)

# Create layer
layer = QuantumLayer(qc, H)
```

## Hybrid Models

You can combine quantum layers with classical layers.

```python
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classical = nn.Linear(2, 2)
        self.quantum = QuantumLayer(qc, H)
        
    def forward(self, x):
        x = self.classical(x)
        x = self.quantum(x)
        return x
```
