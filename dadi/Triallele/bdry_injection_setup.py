from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules = [Extension("bdry_injection",
                 sources=["bdry_injection.pyx"],
                 include_dirs=[np.get_include()])]
)

