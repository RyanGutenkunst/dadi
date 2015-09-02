from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules = [Extension("transition12",
                 sources=["transition12_cy.pyx"],
                 include_dirs=[np.get_include()])]
)

