from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules = [Extension("transition1D",
                 sources=["transition1D.pyx"],
                 include_dirs=[np.get_include()])]
)

