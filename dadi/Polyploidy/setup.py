from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

# Get numpy include directory - allows override via environment variable
numpy_include = os.environ.get('NUMPY_INCLUDE', numpy.get_include())

extensions = [
    Extension(
        "PolyIntegration",  # Just the module name
        sources=[
            "PolyIntegration.pyx",           # Local Cython file
            "integration_shared_poly.c",    # Local polyploid functions
            "../integration_shared.c",      # Parent directory
            "../tridiag.c"                  # Parent directory
        ],
        include_dirs=[
            numpy_include,               # NumPy headers (flexible path)
            ".",                        # Current directory (Polyploids/)
            "..",                       # Parent directory (dadi/)
        ],
        extra_compile_args=["-O3", "-ffast-math"],
        language="c"
    )
]

setup(
    name="PolyIntegration",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'language_level': 3,
        }
    ),
    zip_safe=False,
)