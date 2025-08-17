from setuptools import setup, Extension
from Cython.Build import cythonize

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