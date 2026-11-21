"""Build the native Metal libraries before packaging on macOS."""

import subprocess
import sys
from pathlib import Path

from setuptools import Command, setup
from setuptools.command.build import build


class BuildNative(Command):
    description = "build the Metal library and shaders"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        subprocess.run(["make", "install"], cwd=Path(__file__).parent, check=True)


class Build(build):
    # Editable wheels also run these subcommands, but setuptools can suppress
    # errors from a customized build_py in editable mode.
    sub_commands = [
        ("build_native", lambda cmd: sys.platform == "darwin"),
        *build.sub_commands,
    ]


setup(cmdclass={"build": Build, "build_native": BuildNative})
