"""
result.py - Execution Result
"""
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class MetalQResult:
    """Quantum execution result"""
    
    counts: Dict[str, int]
    shots: int
    num_qubits: int
    num_clbits: int
    
    @classmethod
    def from_json(cls, json_str: str, circuit=None) -> 'MetalQResult':
        data = json.loads(json_str)
        return cls(
            counts=data['counts'],
            shots=data['shots'],
            num_qubits=data['num_qubits'],
            num_clbits=data['num_clbits']
        )
    
    def get_counts(self) -> Dict[str, int]:
        return self.counts.copy()
    
    def get_probabilities(self) -> Dict[str, float]:
        return {k: v / self.shots for k, v in self.counts.items()}
    
    def get_memory(self) -> List[str]:
        memory = []
        for bitstring, count in self.counts.items():
            memory.extend([bitstring] * count)
        return memory
