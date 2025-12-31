"""
metalq/backends/__init__.py - Backend Management

バックエンドの登録、選択、デバイス情報を管理。
"""
from typing import Dict, List, Optional
import platform
import os

from .base import Backend

# Backend registry
_backends: Dict[str, Backend] = {}
_default_device: str = 'auto'


def get_backend(device: str, num_qubits: int = 0) -> Backend:
    """
    Get a backend instance.
    
    Args:
        device: 'cpu', 'mps', or 'auto'
        num_qubits: Number of qubits (for auto selection)
    
    Returns:
        Backend instance
    """
    # Resolve 'auto'
    if device == 'auto':
        device = select_device('auto', num_qubits)
    
    # Get or create backend
    if device not in _backends:
        if device == 'cpu':
            from .cpu.backend import CPUBackend
            _backends['cpu'] = CPUBackend()
        elif device == 'mps':
            from .mps.backend import MPSBackend
            _backends['mps'] = MPSBackend()
        else:
            raise ValueError(f"Unknown device: {device}")
    
    return _backends[device]


def select_device(device: str, num_qubits: int) -> str:
    """
    Select optimal device for given configuration.
    
    Args:
        device: Requested device ('auto', 'cpu', 'mps')
        num_qubits: Number of qubits in circuit
    
    Returns:
        Resolved device name
    """
    if device != 'auto':
        return device
    
    # Check platform
    if platform.system() != 'Darwin':
        return 'cpu'
    
    # Check if MPS is available
    if not _is_mps_available():
        return 'cpu'
    
    # Heuristic: small circuits are faster on CPU (避免 GPU オーバーヘッド)
    # Based on benchmarks: crossover point around 12-14 qubits
    threshold = int(os.environ.get('METALQ_GPU_THRESHOLD', '12'))
    
    if num_qubits <= threshold:
        return 'cpu'
    else:
        return 'mps'


def _is_mps_available() -> bool:
    """Check if Metal GPU backend is available."""
    try:
        from .mps.backend import MPSBackend
        # Try to initialize
        backend = MPSBackend()
        return backend.is_available()
    except Exception:
        return False


def devices() -> List[str]:
    """
    List available devices.
    
    Returns:
        List of device names
    """
    available = ['cpu']  # CPU always available
    
    if platform.system() == 'Darwin' and _is_mps_available():
        available.append('mps')
    
    return available


def device_info(device: Optional[str] = None) -> Dict:
    """
    Get device information.
    
    Args:
        device: Specific device, or None for all
    
    Returns:
        Dict with device information
    """
    info = {
        'platform': platform.system(),
        'processor': platform.processor(),
        'python': platform.python_version(),
    }
    
    # CPU info
    try:
        import multiprocessing
        info['cpu_cores'] = multiprocessing.cpu_count()
    except:
        info['cpu_cores'] = 'unknown'
    
    # Check Numba
    try:
        import numba
        info['numba_version'] = numba.__version__
        info['numba_available'] = True
    except ImportError:
        info['numba_available'] = False
    
    # Check Polars
    try:
        import polars
        info['polars_version'] = polars.__version__
        info['polars_available'] = True
    except ImportError:
        info['polars_available'] = False
    
    # MPS info
    if platform.system() == 'Darwin':
        try:
            from .mps.backend import MPSBackend
            backend = MPSBackend()
            info['mps'] = backend.device_info()
        except Exception as e:
            info['mps'] = {'available': False, 'error': str(e)}
    
    return info


def set_default_device(device: str):
    """
    Set default device for all operations.
    
    Args:
        device: 'cpu', 'mps', or 'auto'
    """
    global _default_device
    if device not in ['cpu', 'mps', 'auto']:
        raise ValueError(f"Invalid device: {device}")
    _default_device = device


def get_default_device() -> str:
    """Get current default device."""
    return _default_device


# Export
__all__ = [
    'Backend',
    'get_backend',
    'select_device',
    'devices',
    'device_info',
    'set_default_device',
    'get_default_device',
]
