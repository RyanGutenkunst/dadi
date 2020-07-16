import os
from pycuda.compiler import SourceModule
import dadi

sourcefile = os.path.join(dadi.__path__[0], "cuda/kernels.cu")
mod = SourceModule(open(sourcefile).read())

_inject_mutations_2D_vals = mod.get_function("inject_mutations_2D")
_inject_mutations_3D_vals = mod.get_function("inject_mutations_3D")
_inject_mutations_4D_vals = mod.get_function("inject_mutations_4D")
_inject_mutations_5D_vals = mod.get_function("inject_mutations_5D")
_Vfunc = mod.get_function("Vfunc")
_Mfunc2D = mod.get_function("Mfunc2D")
_Mfunc3D = mod.get_function("Mfunc3D")
_Mfunc4D = mod.get_function("Mfunc4D")
_Mfunc5D = mod.get_function("Mfunc5D")
_cx0 = mod.get_function("cx0")
_compute_ab_nobc = mod.get_function("compute_ab_nobc")
_compute_bc_nobc = mod.get_function("compute_bc_nobc")
_include_bc = mod.get_function("include_bc")
