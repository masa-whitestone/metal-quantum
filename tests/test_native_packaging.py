"""A failed macOS source build must fail pip and preserve installed shaders."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS native build hook")
@pytest.mark.parametrize("editable", [False, True], ids=["wheel", "editable"])
def test_failed_native_build(tmp_path, editable):
    pytest.importorskip("pip", minversion="21.3")
    pytest.importorskip("setuptools", minversion="64")
    pytest.importorskip("wheel")
    repo = Path(__file__).resolve().parents[1]
    source = tmp_path / "source tree"
    package = source / "metalq"
    binaries = package / "lib"
    binaries.mkdir(parents=True)
    (package / "__init__.py").touch()
    for name in ("setup.py", "pyproject.toml", "Makefile", "README.md", "LICENSE"):
        shutil.copy2(repo / name, source)

    # Keep pip and the root Makefile real. Only the native compiler step fails.
    native = source / "native"
    native.mkdir()
    (native / "Makefile").write_text(
        "all:\n\t@echo simulated-native-build-failure >&2\n\t@exit 1\n"
        "clean:\n\trm -f ../metalq/lib/libmetalq.dylib ../metalq/lib/default.metallib\n"
    )
    names = ("libmetalq.dylib", "default.metallib")
    for name in names:
        (binaries / name).write_bytes(b"previous build")

    output = str(tmp_path / "output")
    action = ["install", "--target", output, "-e"] if editable else ["wheel", "-w", output]
    result = subprocess.run(
        [sys.executable, "-m", "pip", "--disable-pip-version-check",
         *action, ".", "--no-build-isolation", "--no-deps", "--no-cache-dir"],
        cwd=source, env={**os.environ, "PIP_NO_INDEX": "1"},
        capture_output=True, text=True,
    )
    assert "simulated-native-build-failure" in result.stdout + result.stderr
    assert result.returncode != 0
    for name in names:
        assert (binaries / name).read_bytes() == b"previous build"
