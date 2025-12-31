import ctypes
import os

try:
    lib = ctypes.CDLL("./native/build/libmetalq.dylib")
    print("Successfully loaded libmetalq.dylib")
    
    # Check symbol availability
    if hasattr(lib, "metalq_create_context"):
        print("Symbol metalq_create_context found")
    else:
        print("Symbol metalq_create_context NOT found")
        
    lib.metalq_create_context.restype = ctypes.c_void_p
    lib.metalq_destroy_context.argtypes = [ctypes.c_void_p]
        
    # Attempt to create context
    ctx = lib.metalq_create_context()
    print(f"DEBUG: ctx={ctx}")
    if ctx:
        print("Successfully created context")
        lib.metalq_destroy_context(ctx)
        print("Successfully destroyed context")
    else:
        print("Failed to create context")

except Exception as e:
    print(f"Failed to load library: {e}")
