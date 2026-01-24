"""
Cython build setup for pyINLA.

This compiles Python modules to C extensions (.so files) to protect source code.
The compiled wheels contain only binary code - no readable Python source.

Usage:
    # Local build (for testing)
    pip install cython
    python setup.py build_ext --inplace

    # Build wheel
    pip install build cython
    python -m build --wheel
"""

import os
import sys
from pathlib import Path

# Must import setuptools before Cython
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext

# Check if Cython is available
try:
    from Cython.Build import cythonize
    from Cython.Distutils import Extension
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    cythonize = None
    Extension = None


# Package source directory
PACKAGE_DIR = Path(__file__).parent / "install" / "pyinla"

# Python files to compile (exclude __init__.py - it needs to stay as .py for imports)
# Also exclude files with Cython compatibility issues
COMPILE_MODULES = [
    "_api.py",           # Main API - most important to protect
    "_safety.py",        # Safety/validation logic
    "sections.py",       # Section builders
    "models.py",         # Model definitions
    "collect.py",        # Result collection
    "create_data_file.py",
    "control_defaults.py",
    "binary.py",
    "inla_call.py",
    "marginal_utils.py",
    "options.py",
    # "os.py",           # Excluded: potential compatibility issues
    "pyinla_report.py",
    "qinv.py",
    "read_graph.py",
    "rprior.py",
    "scale_model.py",
    "sm.py",
    "surv.py",
    "utils.py",
    # "fmesher_io.py",   # Excluded: has missing import (tempfile)
    "pc_bym.py",
]


def get_extensions():
    """Create Cython extension modules."""
    if not CYTHON_AVAILABLE:
        print("WARNING: Cython not available. Building pure Python package.")
        return []

    extensions = []
    for module_file in COMPILE_MODULES:
        module_path = PACKAGE_DIR / module_file
        if module_path.exists():
            # Convert filename to module name: _api.py -> pyinla._api
            module_name = f"pyinla.{module_file[:-3]}"
            # Use relative path from project root
            relative_path = f"install/pyinla/{module_file}"
            extensions.append(
                Extension(
                    module_name,
                    [relative_path],
                    # Compiler directives for better performance and security
                    cython_directives={
                        'language_level': '3',
                        'embedsignature': False,  # Don't embed signatures (more secure)
                    }
                )
            )

    return cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'embedsignature': False,
        },
        # Don't generate .html annotation files
        annotate=False,
    )


class BuildExtCommand(build_ext):
    """Custom build_ext that removes .py source files after compilation."""

    def run(self):
        super().run()
        # Remove .py source files for compiled modules from build directory
        # This ensures the wheel only contains .so files, not readable source
        if self.build_lib:
            build_pyinla = Path(self.build_lib) / "pyinla"
            if build_pyinla.exists():
                for module_file in COMPILE_MODULES:
                    py_file = build_pyinla / module_file
                    if py_file.exists():
                        print(f"Removing source file: {py_file}")
                        py_file.unlink()


# Custom build_py that excludes compiled modules
from setuptools.command.build_py import build_py

class BuildPyCommand(build_py):
    """Custom build_py that excludes source files for compiled modules."""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        # Filter out modules that will be compiled
        if package == "pyinla":
            compiled_names = {m[:-3] for m in COMPILE_MODULES}  # Remove .py
            modules = [
                (pkg, mod, file)
                for pkg, mod, file in modules
                if mod not in compiled_names
            ]
        return modules


# Only use extensions if Cython is available
ext_modules = get_extensions() if CYTHON_AVAILABLE else []

setup(
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtCommand,
        'build_py': BuildPyCommand,
    } if ext_modules else {},
)
