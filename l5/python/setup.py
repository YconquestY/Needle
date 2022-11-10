"""Setup the package."""
import os
import shutil
import sys
import sysconfig
import platform

from setuptools import find_packages
from setuptools.dist import Distribution

# need to use distutils.core for correct placement of cython dll
if "--inplace" in sys.argv:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension

CURRENT_DIR = os.path.dirname(__file__)

__version__ = "0.1.0"


def config_pybind():
    ret = []
    extra_compile_args = ["-std=c++14"]

    if os.name == "nt":
        library_dirs = ["needle", "../build/Release", "../build"]
        libraries = ["needle"]
        extra_compile_args = None
    else:
        library_dirs = None
        libraries = None

    ret.append(
        Extension(
            "needle._ffi.main",
            ["pybind/main.cc"],
            include_dirs=[
                "../include",
                "../3rdparty/dlpack/include",
                "../3rdparty/pybind11/include",
            ],
            extra_compile_args=extra_compile_args,
            library_dirs=library_dirs,
            libraries=libraries,
            language="c++",
        )
    )
    return ret


setup(
    name="needle",
    version=__version__,
    description="DLsys",
    zip_safe=False,
    packages=find_packages(),
    package_dir={"needle": "needle"},
    url="dlsyscourse.org",
    ext_modules=config_pybind(),
)
