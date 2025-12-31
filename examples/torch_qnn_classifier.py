"""
examples/torch_qnn_classifier.py - Quantum Neural Network Classifier

This example demonstrates how to integrate Metal-Q with PyTorch to build a Quantum Neural Network (QNN).
We perform binary classification on a synthetic dataset.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# 1. Dataset Generation (Moons)
def get_data(n_samples=200):
    try:
        from sklearn.datasets import make_moons
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    except ImportError:
        print("Scikit-learn not found. Using simple synthetic data.")
        # XOR problem
        torch.manual_seed(42)
        X = torch.rand(n_samples, 2) * 2 - 1 # [-1, 1]
        # x0*x1 > 0 -> 1, else 0 (XOR-like quadrants)
        y = ((X[:, 0] * X[:, 1]) > 0).float()
        
        # Split
        split = int(n_samples * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

    return torch.tensor(X_train, dtype=torch.float32), \
           torch.tensor(X_test, dtype=torch.float32), \
           torch.tensor(y_train, dtype=torch.float32), \
           torch.tensor(y_test, dtype=torch.float32)


class QNNClassifier(nn.Module):
    def __init__(self, n_qubits=2, n_layers=2):
        super().__init__()
        
        self.n_qubits = n_qubits
        
        # Define Ansatz
        # We need a way to encode data 'x' AND have trainable params 'theta'.
        # MetalQ 'QuantumLayer' binds everything trainable.
        # To handle data encoding, we usually define a circuit with SOME params intended for data.
        # But `QuantumLayer` treats ALL parameters in circuit as weights.
        # WORKAROUND: MetalQ currently doesn't have explicit "Data Re-uploading" utility layer.
        # We can simulate it by prepending encoding layers in a loop or using Torch logic.
        
        # Actually, `QuantumLayer` implementation in `metalq/torch/layer.py`:
        # `__init__`: `self.weights = nn.Parameter(...)` for ALL circuit parameters.
        # `forward()`: `circuit.bind_parameters({p: self.weights[i]})`.
        # It does NOT accept input `x` to bind to specific params.
        # It takes no args in forward? `def forward(self):`.
        # Check `metalq/torch/layer.py`.
        
        # If `QuantumLayer` takes no input, it acts as a trainable constant generator (like VQE).
        # To make a Classifier (QNN), we need `forward(x)`.
        # We need to modify `QuantumLayer` or subclass it, or create a new `QNNLayer`.
        
        # For this example, I will implement a custom `HybridQNN` that manages binding manually.
        # Or better, show how to use `metalq.torch.function.QuantumFunction` directly.
            
        pass
        
    def build_circuit(self, inputs, weights):
        # We can't build circuit dynamically in forward pass efficiently.
        # We define a circuit with Parameters 'x' and 'theta'.
        pass

# Let's verify `metalq/torch/layer.py` capability first.
# Step 1087 showed `loss = layer()`. No args.
# So `QuantumLayer` is VQE-only currently.

# I will implement a `DataEncodingLayer` in this example script to demonstrate how to extend MetalQ.

class QCPLayer(nn.Module):
    """
    Quantum Classification Layer using Data Re-uploading or Angle Encoding.
    """
    def __init__(self, n_qubits, n_layers, backend='mps'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = backend
        
        # Define Circuit structure ONCE
        self.circuit = Circuit(n_qubits)
        self.params_input = []   # Parameters for X
        self.params_weight = []  # Parameters for Weights
        
        # Universal Ansatz with Data Encoding
        # Layer: H -> RZ(x_i) -> RY(theta_i) -> Entangle
        
        # Input features: 2 (x0, x1).
        # We map x0->q0, x1->q1.
        
        for i in range(n_qubits):
            self.circuit.h(i)
            
        for l in range(n_layers):
            # Data Encoding (RZ)
            for q in range(n_qubits):
                p_in = Parameter(f'x_{l}_{q}')
                self.circuit.rz(p_in, q) # Rotation depends on input
                self.params_input.append(p_in)
                
            # Trainable (RY, RZ)
            for q in range(n_qubits):
                p_w1 = Parameter(f'w1_{l}_{q}')
                p_w2 = Parameter(f'w2_{l}_{q}')
                self.circuit.ry(p_w1, q)
                self.circuit.rz(p_w2, q)
                self.params_weight.append(p_w1)
                self.params_weight.append(p_w2)
                
            # Entangling
            if n_qubits > 1:
                for q in range(n_qubits-1):
                    self.circuit.cx(q, q+1)
                    
        # Measurement: Z0
        self.hamiltonian = Z(0)
        
        # Torch Weights
        # Initialize randomly
        self.weights = nn.Parameter(torch.rand(len(self.params_weight)) * 2 * np.pi)
        
    def forward(self, x):
        """
        Forward pass.
        x: (batch_size, n_features). n_features must match n_qubits used for encoding (2).
        Returns: (batch_size, 1) expectation values.
        """
        # MetalQ currently doesn't support Batched Execution in backend efficiently
        # (It loops over batch if implemented, or we loop here).
        # `QuantumFunction` supports torch autograd.
        
        from metalq.torch.function import QuantumFunction
        
        # We need to construct the full parameter tensor for each batch item.
        # [x_params, w_params]
        # x: (batch, n_inputs). weights: (n_weights).
        # We need to map them correctly to `self.circuit.parameters`.
        # `QuantumFunction` takes `input_tensor` (concatenated params) and `circuit`, `Hamiltonian`.
        # It binds `input_tensor[i]` to `circuit.parameters[i]`.
        # So we must ensure concatenation order matches `circuit.parameters` order.
        
        # Sort circuit parameters
        ordered_params = self.circuit.parameters # Sorted by name
        # print(f"DEBUG QNN: ordered_params len={len(ordered_params)}")
        
        # We need to construct a tensor of shape (batch, total_params)
        # where columns correspond to ordered_params.
        
        # Map param -> index in ordered_params
        p_to_idx = {p: i for i, p in enumerate(ordered_params)}
        
        batch_size = x.shape[0]
        total_p = len(ordered_params)
        full_params = torch.zeros(batch_size, total_p)
        # print(f"DEBUG QNN: full_params shape={full_params.shape}")

        
        # Fill Weights (constant for batch)
        # We need to know which index in `full_params` corresponds to `self.weights[k]`.
        # Precompute indices
        if not hasattr(self, 'indices_map'):
             self.indices_map = []
             # For each param in ordered_params:
             # Is it Input? Which col?
             # Is it Weight? Which index?
             for idx, p in enumerate(ordered_params):
                 if p.name.startswith('x'):
                     # name x_{l}_{q}
                     parts = p.name.split('_')
                     q_idx = int(parts[2])
                     self.indices_map.append(('x', q_idx))
                 elif p.name.startswith('w'):
                     # find index in self.params_weight
                     # inefficient search but init only
                     try:
                         w_idx = self.params_weight.index(p)
                         self.indices_map.append(('w', w_idx))
                     except:
                         pass
        
        # Fill
        for i in range(total_p):
            itype, idx = self.indices_map[i]
            if itype == 'x':
                full_params[:, i] = x[:, idx]
            else:
                full_params[:, i] = self.weights[idx]
                
        # Run Quantum Function
        # MetalQ MPS backend is single-shot, so we loop over batch.
        # QuantumFunction expects 1D tensor of parameters.
        outs = []
        for i in range(batch_size):
            p_i = full_params[i] # 1D tensor
            # Return is scalar tensor
            out = QuantumFunction.apply(p_i, self.circuit, self.hamiltonian, self.backend)
            outs.append(out)
            
        return torch.stack(outs) # (batch_size,)



def main():
    print("=== Metal-Q PyTorch QNN Example ===")
    
    # Check dependencies
    try:
        import sklearn
    except ImportError:
        print("Scikit-learn not found. Please install it to run this example.")
        return

    # Data
    X_train, X_test, y_train, y_test = get_data()
    print(f"Data: {len(X_train)} train, {len(X_test)} test samples.")
    
    # Model
    model = QCPLayer(n_qubits=2, n_layers=2, backend='mps')
    
    # Training Setup
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss() # Label 0/1 -> Target -1/1 expectation?
    # Map y: 0 -> -1, 1 -> 1.
    y_train_mapped = 2 * y_train - 1
    y_test_mapped = 2 * y_test - 1
    
    print("\nStarting Training (5 Epochs)...")
    
    for epoch in range(5):
        total_loss = 0
        optimizer.zero_grad()
        
        # Single batch (all data)
        # For speed in MPS simulation, small batch is better but let's try full.
        # MetalQ backend executes sequentially currently (no batching opt).
        
        preds = model(X_train)
        loss = loss_fn(preds.squeeze(), y_train_mapped)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
        
    print("\nTraining Complete.")
    
    # Evaluate
    with torch.no_grad():
        test_preds = model(X_test).squeeze()
        # Threshold at 0
        predictions = (test_preds > 0).float()
        # y_test is 0/1 parameter
        acc = accuracy_score(y_test.numpy(), predictions.numpy())
        
    print(f"Test Accuracy: {acc * 100:.2f}%")
    
    if acc > 0.7:
        print("SUCCESS: Classifier learned something.")
    else:
        print("WARNING: Low accuracy. Adjust hyperparameters?")

if __name__ == "__main__":
    main()
