#!/usr/bin/env python3
"""
Setup script for Metal-Q.
Handles building native libraries during installation.
"""

import subprocess
import sys
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop


def build_native():
    """Build native Metal libraries."""
    if sys.platform != 'darwin':
        print("Warning: Metal-Q only supports macOS. Skipping native build.")
        return False
    
    try:
        subprocess.run(['xcrun', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Xcode Command Line Tools required.")
        print("Install with: xcode-select --install")
        return False
    
    root = Path(__file__).parent
    print("Building Metal-Q native libraries...")
    
    try:
        subprocess.run(['make', 'clean'], cwd=root, capture_output=True)
        subprocess.run(['make', 'install'], cwd=root, check=True)
        print("Native libraries built successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        return False


class BuildPy(build_py):
    def run(self):
        build_native()
        super().run()


class Develop(develop):
    def run(self):
        build_native()
        super().run()


setup(
    cmdclass={
        'build_py': BuildPy,
        'develop': Develop,
    },
)
