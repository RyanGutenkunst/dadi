from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "Int3D_poly",  # Just the module name
        sources=[
            "Int3D_poly.pyx",           # Local Cython file
            "integration_shared_poly.c", # Local polyploid functions
            "../integration_shared.c",   # Parent directory
            "../tridiag.c"              # Parent directory
        ],
        include_dirs=[
            numpy.get_include(),
            ".",                        # Current directory (Polyploids/)
            "..",                       # Parent directory (dadi/)
        ],
        extra_compile_args=["-O3", "-ffast-math"],
        language="c"
    )
]

setup(
    name="Int3D_poly",
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