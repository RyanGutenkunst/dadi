import os
from pycuda.compiler import SourceModule
import dadi

sourcefile = os.path.join(dadi.__path__[0], "cuda/kernels.cu")
mod = SourceModule(open(sourcefile).read())
_inject_mutations_2D_vals = mod.get_function("inject_mutations_2D")