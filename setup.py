"""The C extension, which pyproject.toml cannot declare: cffi compiles csrc/ at build time."""

from setuptools import setup

setup(cffi_modules=["build_fsim.py:ffibuilder"])
