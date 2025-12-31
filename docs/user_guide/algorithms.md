# Algorithms

Metal-Q comes with built-in implementations of popular quantum algorithms.

## VQE (Variational Quantum Eigensolver)

Used for finding the ground state energy of a Hamiltonian.

```python
from metalq.algorithms import VQE
# ... define hamiltonian and ansatz ...
vqe = VQE(ansatz, hamiltonian)
result = vqe.compute_minimum_eigenvalue()
```

## QAOA (Quantum Approximate Optimization Algorithm)

Used for solving combinatorial optimization problems like MaxCut.

```python
from metalq.algorithms import QAOA
# ... define cost hamiltonian ...
qaoa = QAOA(cost_hamiltonian, p=2)
result = qaoa.solve()
```
