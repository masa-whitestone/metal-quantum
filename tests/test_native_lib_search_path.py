"""The native library must never be loaded from the current directory (CWE-427).

``ctypes.CDLL`` runs a dylib's initializers as soon as it is opened, so a
``native/build/libmetalq.dylib`` planted in an untrusted working directory
must not be picked up. Only the copy shipped in the package (resolved from
``__file__``) is loaded, unless ``METALQ_NATIVE_LIB`` names an absolute path.
"""

import ctypes
import logging
import os
import shutil
import stat
import sys
from unittest.mock import MagicMock

import pytest

import metalq
from metalq.backends.mps import backend as mps_backend
from metalq.backends.mps.backend import MPSBackend
from metalq.exceptions import LibraryLoadError

LIB = mps_backend.LIB_NAME
METALLIB = mps_backend.METALLIB_NAME
ENV = mps_backend.NATIVE_LIB_ENV_VAR
LOGGER = mps_backend.__name__


def _write_pair(directory, payload=b"fake"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / LIB).write_bytes(payload)
    (directory / METALLIB).write_bytes(payload)
    return directory / LIB


def _fake_lib():
    lib = MagicMock()
    lib.metalq_create_context = MagicMock(return_value=ctypes.c_void_p(123))
    lib.metalq_destroy_context = MagicMock(return_value=0)
    lib.metalq_is_supported = MagicMock(return_value=True)
    return lib


@pytest.fixture
def clean_env(monkeypatch):
    for name in (ENV, "METALQ_DOCS_BUILD", "READTHEDOCS"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def cdll_calls(monkeypatch):
    """Record every path handed to ctypes.CDLL without opening anything."""
    calls = []

    def fake_cdll(path, *args, **kwargs):
        calls.append(path)
        return _fake_lib()

    monkeypatch.setattr(mps_backend.ctypes, "CDLL", fake_cdll)
    return calls


@pytest.fixture
def attacker_cwd(tmp_path, monkeypatch):
    """An untrusted working directory with planted libraries in it."""
    cwd = tmp_path / "untrusted project"
    _write_pair(cwd / "native" / "build", b"planted")
    _write_pair(cwd, b"planted")
    _write_pair(cwd / "lib", b"planted")
    monkeypatch.chdir(cwd)
    return cwd


@pytest.fixture
def packaged_dir(tmp_path, monkeypatch):
    """Stand-in for the installed package's metalq/lib directory."""
    directory = tmp_path / "site-packages" / "metalq" / "lib"
    _write_pair(directory, b"packaged")
    monkeypatch.setattr(mps_backend, "_PACKAGE_LIB_DIR", str(directory))
    return directory


def _new_backend_without_init():
    return MPSBackend.__new__(MPSBackend)


def test_package_lib_dir_is_absolute_and_based_on_file(attacker_cwd):
    expected = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(metalq.__file__)), "lib")
    )
    assert os.path.isabs(mps_backend._PACKAGE_LIB_DIR)
    assert mps_backend._PACKAGE_LIB_DIR == expected
    assert not mps_backend._PACKAGE_LIB_DIR.startswith(str(attacker_cwd))


def test_cwd_native_build_is_ignored_packaged_copy_loaded(
        clean_env, cdll_calls, attacker_cwd, packaged_dir):
    MPSBackend()
    assert cdll_calls == [str(packaged_dir / LIB)]


def test_cwd_native_build_is_ignored_without_packaged_copy(
        clean_env, cdll_calls, attacker_cwd, tmp_path, monkeypatch):
    monkeypatch.setattr(mps_backend, "_PACKAGE_LIB_DIR", str(tmp_path / "empty"))
    with pytest.raises(LibraryLoadError):
        MPSBackend()
    assert cdll_calls == []


def test_loaded_path_is_logged_at_debug(
        clean_env, cdll_calls, attacker_cwd, packaged_dir, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    _new_backend_without_init()._load_library()
    messages = [r.getMessage() for r in caplog.records
                if r.name == LOGGER and r.levelno == logging.DEBUG]
    assert any(str(packaged_dir / LIB) in m for m in messages)


@pytest.mark.parametrize("value", [
    os.path.join("native", "build", LIB),
    LIB,
    os.path.join(".", LIB),
    os.path.join("~", LIB),
])
def test_relative_override_is_rejected(
        clean_env, cdll_calls, attacker_cwd, packaged_dir, monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    with pytest.raises(LibraryLoadError, match="absolute"):
        _new_backend_without_init()._load_library()
    assert cdll_calls == []


def test_absolute_override_is_loaded(
        clean_env, cdll_calls, attacker_cwd, packaged_dir, tmp_path, monkeypatch):
    dev_build = _write_pair(tmp_path / "dev" / "native" / "build", b"dev")
    monkeypatch.setenv(ENV, str(dev_build))
    _new_backend_without_init()._load_library()
    assert cdll_calls == [str(dev_build)]


def test_empty_override_means_packaged_copy(
        clean_env, cdll_calls, attacker_cwd, packaged_dir, monkeypatch):
    monkeypatch.setenv(ENV, "")
    _new_backend_without_init()._load_library()
    assert cdll_calls == [str(packaged_dir / LIB)]


def test_missing_override_fails_instead_of_falling_back(
        clean_env, cdll_calls, attacker_cwd, packaged_dir, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, str(tmp_path / "nowhere" / LIB))
    with pytest.raises(LibraryLoadError):
        _new_backend_without_init()._load_library()
    assert cdll_calls == []


def test_override_without_adjacent_metallib_is_rejected(
        clean_env, cdll_calls, attacker_cwd, packaged_dir, tmp_path, monkeypatch):
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    (lonely / LIB).write_bytes(b"dev")
    monkeypatch.setenv(ENV, str(lonely / LIB))
    with pytest.raises(LibraryLoadError, match=METALLIB):
        _new_backend_without_init()._load_library()
    assert cdll_calls == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_world_writable_library_is_refused(
        clean_env, cdll_calls, attacker_cwd, packaged_dir):
    lib = packaged_dir / LIB
    lib.chmod(lib.stat().st_mode | stat.S_IWOTH)
    with pytest.raises(LibraryLoadError):
        _new_backend_without_init()._load_library()
    assert cdll_calls == []


@pytest.mark.skipif(sys.platform != "darwin", reason="needs a real libmetalq.dylib")
def test_real_dylib_planted_in_cwd_is_not_loaded(clean_env, tmp_path, monkeypatch):
    """End to end with real binaries: a working copy in CWD must lose."""
    packaged = os.path.join(mps_backend._PACKAGE_LIB_DIR, LIB)
    if not (os.path.isfile(packaged)
            and os.path.isfile(os.path.join(mps_backend._PACKAGE_LIB_DIR, METALLIB))):
        pytest.skip("packaged native library not built")

    cwd = tmp_path / "untrusted project"
    planted = cwd / "native" / "build"
    planted.mkdir(parents=True)
    for name in (LIB, METALLIB):
        shutil.copy2(os.path.join(mps_backend._PACKAGE_LIB_DIR, name), planted / name)
    monkeypatch.chdir(cwd)

    # Let the backend be released normally: calling __del__() by hand as well
    # would destroy the native context twice.
    backend = MPSBackend()
    loaded = backend._lib._name
    del backend
    assert os.path.samefile(loaded, packaged)
    assert not os.path.samefile(loaded, planted / LIB)
