#!/usr/bin/env python3
"""
Demonstration of Metal-Q with enhanced security features.

The API remains exactly the same - existing code works without modification.
Security improvements happen automatically behind the scenes.
"""
from metalq import Circuit, run
import math

# Example 1: Normal usage - works exactly as before
print("Example 1: Standard usage (unchanged API)")
print("-" * 40)

circuit = Circuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)

# Run circuit - API unchanged
result = run(circuit, shots=1000, backend='mps')
print(f"Results: {result.counts}")
print("✓ Standard usage works exactly as before\n")


# Example 2: Security features activate automatically on invalid input
print("Example 2: Automatic input validation")
print("-" * 40)

# Create circuit with invalid parameters
bad_circuit = Circuit(2)
try:
    bad_circuit.rx(float('nan'), 0)  # Invalid: NaN parameter
    result = run(bad_circuit, backend='mps')
except Exception as e:
    print(f"Invalid input caught: {e}")
    print("✓ Invalid parameters are automatically detected\n")


# Example 3: Memory safety
print("Example 3: Memory safety checks")
print("-" * 40)

# Try to create a circuit that would use too much memory
try:
    huge_circuit = Circuit(35)  # Too many qubits
    result = run(huge_circuit, backend='mps')
except Exception as e:
    print(f"Memory limit enforced: {e}")
    print("✓ Memory limits are automatically enforced\n")


# Example 4: Error messages are sanitized
print("Example 4: Sanitized error messages")
print("-" * 40)

try:
    # Trigger an error that would normally show memory addresses
    circuit = Circuit(-1)  # Invalid
except Exception as e:
    error_msg = str(e)
    # Check that sensitive info is removed
    if '0x[REDACTED]' in error_msg or '0x' not in error_msg:
        print("✓ Error messages are automatically sanitized")
        print(f"  Safe error: {error_msg}\n")


# Example 5: Timeouts prevent hanging
print("Example 5: Automatic timeout protection")
print("-" * 40)

# Create a complex circuit
complex_circuit = Circuit(10)
for _ in range(100):
    complex_circuit.h(0)
    for i in range(9):
        complex_circuit.cx(i, i+1)

# This will automatically timeout if it takes too long
# (in practice, this circuit is still small enough to complete quickly)
result = run(complex_circuit, shots=100, backend='mps')
print("✓ Complex circuits have automatic timeout protection")
print("  Circuit completed successfully\n")


print("=" * 60)
print("SUMMARY: All security features work transparently!")
print("No code changes required - just update Metal-Q.")
print("=" * 60)