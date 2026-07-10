"""
tests/_circuit_helpers.py - Shared random-circuit builders for fusion tests

test_fusion.py / test_fusion_extras.py で共有するゲート名定数と
ランダム回路生成器。rng は呼び出し側が渡す (ファイルごとの seed 独立性
を保つため)。
"""
import numpy as np

SINGLE = ['x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'sx']
SINGLE_P = ['rx', 'ry', 'rz', 'p']
TWO = ['cx', 'cy', 'cz', 'ch', 'swap', 'iswap']
TWO_P = ['crx', 'cry', 'crz', 'cp', 'rxx', 'ryy', 'rzz', 'rzx']
THREE = ['ccx', 'ccz', 'cswap']


def random_circuit(rng, n, depth):
    gates = []
    for _ in range(depth):
        r = rng.random()
        if n < 3 and r >= 0.9:
            r = 0.5
        if r < 0.35:
            gates.append({'name': str(rng.choice(SINGLE)),
                          'qubits': [int(rng.integers(n))], 'params': []})
        elif r < 0.6:
            gates.append({'name': str(rng.choice(SINGLE_P)),
                          'qubits': [int(rng.integers(n))],
                          'params': [float(rng.uniform(-np.pi, np.pi))]})
        elif r < 0.75:
            q = rng.choice(n, size=2, replace=False)
            gates.append({'name': str(rng.choice(TWO)),
                          'qubits': [int(q[0]), int(q[1])], 'params': []})
        elif r < 0.9:
            q = rng.choice(n, size=2, replace=False)
            gates.append({'name': str(rng.choice(TWO_P)),
                          'qubits': [int(q[0]), int(q[1])],
                          'params': [float(rng.uniform(-np.pi, np.pi))]})
        else:
            q = rng.choice(n, size=3, replace=False)
            gates.append({'name': str(rng.choice(THREE)),
                          'qubits': [int(q[0]), int(q[1]), int(q[2])],
                          'params': []})
    return gates
