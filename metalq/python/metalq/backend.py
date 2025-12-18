"""
backend.py - Metal-Q Backend
"""
import ctypes
import os
import json
from pathlib import Path
from typing import Optional, Union, Dict
from qiskit import QuantumCircuit
import numpy as np

from .circuit_data import circuit_to_json
from .result import MetalQResult


class MetalQBackend:
    """Quantum simulator backend using Metal GPU"""
    
    _instance: Optional['MetalQBackend'] = None
    _lib: Optional[ctypes.CDLL] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize native library"""
        try:
            lib_path = self._find_library()
            self._lib = ctypes.CDLL(lib_path)
            self._setup_functions()
            
            result = self._lib.metalq_init()
            if result != 0:
                raise RuntimeError(f"Failed to initialize Metal-Q: error code {result}")
        except FileNotFoundError:
            # Allow instantiation even if lib not found (e.g. during pdoc generation or initial setup)
            # but methods will fail
            self._lib = None
    
    def _find_library(self) -> str:
        """Find libmetalq.dylib"""
        # Priority: 
        # 1. 'lib' subdirectory in the package (runtime)
        # 2. Build directory (dev)
        
        package_lib = Path(__file__).parent / 'lib' / 'libmetalq.dylib'
        if package_lib.exists():
            return str(package_lib)
            
        # Development path relative to this file
        # metalq/python/metalq/backend.py -> root/build/libmetalq.dylib
        dev_build_lib = Path(__file__).parent.parent.parent.parent / 'build' / 'libmetalq.dylib'
        if dev_build_lib.exists():
            return str(dev_build_lib)

        raise FileNotFoundError(
            "libmetalq.dylib not found. "
            "Please build the native library first using 'make install'."
        )
    
    def _setup_functions(self):
        """Setup C function signatures"""
        if not self._lib: return

        # int metalq_init(void)
        self._lib.metalq_init.restype = ctypes.c_int
        self._lib.metalq_init.argtypes = []
        
        # int metalq_run_circuit(const char*, int, char**, int*)
        self._lib.metalq_run_circuit.restype = ctypes.c_int
        self._lib.metalq_run_circuit.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_int)
        ]
        
        # void metalq_free_result(char*)
        self._lib.metalq_free_result.restype = None
        self._lib.metalq_free_result.argtypes = [ctypes.c_char_p]
        
        # void metalq_cleanup(void)
        self._lib.metalq_cleanup.restype = None
        self._lib.metalq_cleanup.argtypes = []
        
        # int metalq_get_statevector(const char*, float**, float**, int*)
        self._lib.metalq_get_statevector.restype = ctypes.c_int
        self._lib.metalq_get_statevector.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_int)
        ]
        
        # void metalq_free_statevector(float*, float*)
        self._lib.metalq_free_statevector.restype = None
        self._lib.metalq_free_statevector.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]
    
    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024
    ) -> MetalQResult:
        if not self._lib:
            raise RuntimeError("Metal-Q native library not loaded.")
            
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("circuit must be a Qiskit QuantumCircuit")
        
        if not 1 <= shots <= 8192:
            raise ValueError("shots must be between 1 and 8192")
        
        circuit_json = circuit_to_json(circuit)
        
        result_ptr = ctypes.c_char_p()
        result_len = ctypes.c_int()
        
        status = self._lib.metalq_run_circuit(
            circuit_json.encode('utf-8'),
            shots,
            ctypes.byref(result_ptr),
            ctypes.byref(result_len)
        )
        
        if status != 0:
            raise RuntimeError(f"Simulation failed with error code: {status}")
        
        result_json = result_ptr.value.decode('utf-8')
        self._lib.metalq_free_result(result_ptr)
        
        return MetalQResult.from_json(result_json, circuit)
    
    def statevector(self, circuit: QuantumCircuit):
        if not self._lib:
            raise RuntimeError("Metal-Q native library not loaded.")
            
        # Create a copy without final measurements for statevector calculation
        # Note: remove_final_measurements is not always available or perfect in all versions, 
        # so we rely on the user passing a circuit or we handle it by not processing measurement gates in C++ for statevector op
        # simpler approach: Just pass it. The C++ statevector function will just execute gates.
        
        circuit_json = circuit_to_json(circuit)
        
        real_ptr = ctypes.POINTER(ctypes.c_float)()
        imag_ptr = ctypes.POINTER(ctypes.c_float)()
        length = ctypes.c_int()
        
        status = self._lib.metalq_get_statevector(
            circuit_json.encode('utf-8'),
            ctypes.byref(real_ptr),
            ctypes.byref(imag_ptr),
            ctypes.byref(length)
        )
        
        if status != 0:
            raise RuntimeError(f"Statevector computation failed: {status}")
        
        n = length.value
        real_arr = np.ctypeslib.as_array(real_ptr, shape=(n,)).copy()
        imag_arr = np.ctypeslib.as_array(imag_ptr, shape=(n,)).copy()
        
        self._lib.metalq_free_statevector(real_ptr, imag_ptr)
        
        return real_arr + 1j * imag_arr
    
    def __del__(self):
        if self._lib:
            pass
            # Ideally call cleanup, but it might be unsafe during interpreter shutdown
            # self._lib.metalq_cleanup()

_backend: Optional[MetalQBackend] = None

def get_backend() -> MetalQBackend:
    global _backend
    if _backend is None:
        _backend = MetalQBackend()
    return _backend
