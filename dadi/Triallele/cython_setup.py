from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    cmdclass={'build_ext': build_ext},
    include_dirs = [np.get_include()],
    ext_modules = [Extension("transition1", sources=["transition1.pyx"]),
                   Extension("transition2", sources=["transition2.pyx"]),
                   Extension("transition12", sources=["transition12.pyx"]),
                   Extension("bdry_injection", sources=["bdry_injection.pyx"]),
                   Extension("transition1D", sources=["transition1D.pyx"]),
                   ]
)

