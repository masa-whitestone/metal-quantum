
import metalq as mq

def test_api():
    print("Testing High-Level API...")
    
    # Circuit
    qc = mq.Circuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    # Run
    print("Running circuit...")
    res = mq.run(qc, shots=0, backend='mps')
    print("Statevector:", res['statevector'])
    
    # Expectation
    H = mq.Z(0) * mq.Z(1)
    val = mq.expect(qc, H, backend='mps')
    print(f"Expectation <ZZ>: {val}")
    
    # Check Bell State <ZZ> should be 1.0 (since |00> + |11> -> Z(0)Z(1) = 1)
    if abs(val - 1.0) < 1e-4:
        print("SUCCESS: API Expectation correct.")
    else:
        print("FAILURE: API Expectation incorrect.")

if __name__ == "__main__":
    test_api()
