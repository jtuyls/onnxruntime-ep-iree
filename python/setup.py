"""Setup script for the iree-onnx-ep Python package.

This package wraps a native shared library (the IREE Execution Provider plugin
for ONNX Runtime) and provides Python helpers to locate it at runtime.

Design decisions
================

Environment variable ``IREE_ONNX_EP_BUILD_DIR``
    The native library is built externally via CMake, not by this setup script.
    This environment variable tells setup.py where the pre-built library lives.
    It is only required at *install* time, NOT at runtime.

    For editable installs (``pip install -e``), setup.py only validates that the
    library exists in the build directory — it does NOT copy the library or
    generate any files.  At runtime, ``get_library_path()`` finds the library
    by walking up from the package's source location to ``<project_root>/build/``.
    This means rebuilds are picked up immediately without reinstalling.

    For wheel builds (``pip wheel`` / ``pip install``), setup.py copies the
    library into the package directory so it is bundled in the wheel.

No generated files in the source tree
    Unlike many packaging setups, we deliberately avoid writing any files into
    the source tree (no copied libraries, no config files).  This keeps the
    working directory clean for git.

``BinaryDistribution``
    We override ``has_ext_modules()`` to return True.  This forces setuptools to
    generate platform-specific wheels (e.g. ``linux_x86_64``, ``macosx_arm64``)
    instead of a universal ``py3-none-any`` wheel.  This is necessary because the
    package contains a native shared library.

``package_data``
    Includes ``*.dylib``, ``*.so``, ``*.dll`` so that when the library IS copied
    for wheel builds, it ends up in the installed package.

Usage
=====

    # Editable install (development)
    IREE_ONNX_EP_BUILD_DIR=/path/to/build uv pip install -e .

    # Regular install / wheel build
    IREE_ONNX_EP_BUILD_DIR=/path/to/build uv pip install .
"""

import os
import pathlib
import shutil

from setuptools import Distribution, setup

script_dir = pathlib.Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Locate the pre-built native library via environment variable.
# ---------------------------------------------------------------------------
build_dir = os.environ.get("IREE_ONNX_EP_BUILD_DIR")
if not build_dir:
    raise EnvironmentError(
        "IREE_ONNX_EP_BUILD_DIR environment variable must be set to the "
        "directory containing the built libiree_onnx_ep shared library."
    )

build_dir = pathlib.Path(build_dir).resolve()

# Find the library in the build directory.
lib_names = ["libiree_onnx_ep.dylib", "libiree_onnx_ep.so", "iree_onnx_ep.dll"]
found = [build_dir / name for name in lib_names if (build_dir / name).exists()]
if len(found) != 1:
    raise FileNotFoundError(
        f"Expected exactly one EP library in {build_dir}, "
        f"found: {[str(p) for p in found]}"
    )
ep_lib_path = found[0]

# ---------------------------------------------------------------------------
# For non-editable installs, copy the library into the package so it gets
# bundled into the wheel.  For editable installs this copy is unnecessary
# (get_library_path() finds it via relative path), but it's harmless and
# keeps setup.py simple — no need to detect which install mode we're in.
# ---------------------------------------------------------------------------
pkg_dir = script_dir / "iree_onnx_ep"
dst = pkg_dir / ep_lib_path.name
if dst.resolve() != ep_lib_path.resolve():
    shutil.copyfile(ep_lib_path, dst)


# ---------------------------------------------------------------------------
# Setuptools configuration.
# ---------------------------------------------------------------------------
class BinaryDistribution(Distribution):
    """Mark the distribution as containing native code."""

    def has_ext_modules(self):
        return True


setup(
    name="iree-onnx-ep",
    version="0.1.0",
    description="Python helpers for the IREE ONNX Runtime Execution Provider",
    packages=["iree_onnx_ep"],
    package_data={
        "iree_onnx_ep": ["*.dylib", "*.so", "*.dll"],
    },
    include_package_data=True,
    distclass=BinaryDistribution,
    python_requires=">=3.10",
)
