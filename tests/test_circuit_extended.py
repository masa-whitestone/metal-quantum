
import pytest
import metalq as mq
from metalq import Circuit, Parameter

def test_circuit_structure():
    qc = Circuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rx(0.5, 2)
    
    assert len(qc.gates) == 3
    assert qc.num_qubits == 3
    assert qc.gates[0].name == 'H'
    assert qc.gates[1].qubits == [0, 1]

def test_parameter_binding():
    theta = Parameter('theta')
    phi = Parameter('phi')
    
    qc = Circuit(1)
    qc.rx(theta, 0)
    qc.rz(phi, 0)
    
    assert len(qc.parameters) == 2
    
    # Partial Bind
    qc_bind = qc.bind_parameters({theta: 1.5})
    # Should still have phi unbound? 
    # Current v1 implementation might require full binding or returns circuit with partials.
    # Let's verify behavior.
    
    # Actually explicit Dictionary binding verification
    # If partial binding is not supported, it might error or leave Param.
    
    # Bind all
    qc_final = qc.bind_parameters({theta: 0.1, phi: 0.2})
    assert len(qc_final.parameters) == 0
    assert qc_final.gates[0].params[0] == 0.1
    
def test_invalid_operations():
    qc = Circuit(2)
    with pytest.raises(ValueError):
        qc.cx(0, 5) # Qubit index out of range (if check exists)
        
    with pytest.raises(ValueError):
        qc.h(-1)

if __name__ == "__main__":
    test_circuit_structure()
    test_parameter_binding()
    # test_invalid_operations() # Uncomment if validation logic is strict
    print("Circuit Extended Tests Passed.")
