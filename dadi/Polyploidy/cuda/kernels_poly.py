import os
from pycuda.compiler import SourceModule
import dadi

# TODO: Check if this is the correct file path
sourcefile = os.path.join(dadi.__path__[0], "Polyploidy/cuda/kernels_poly.cu")
mod = SourceModule(open(sourcefile).read())

# general functions - mutation injection, memory management, and a,b,c matrix computation
_inject_mutations_2D_vals = mod.get_function("inject_mutations_2D")
_inject_mutations_3D_vals = mod.get_function("inject_mutations_3D")
_inject_mutations_4D_vals = mod.get_function("inject_mutations_4D")
_inject_mutations_5D_vals = mod.get_function("inject_mutations_5D")
_cx0 = mod.get_function("cx0")
_compute_ab_nobc = mod.get_function("compute_ab_nobc")
_compute_bc_nobc = mod.get_function("compute_bc_nobc")

# variance functions
_Vfunc = mod.get_function("Vfunc")
_Vfunc_tetra = mod.get_function("Vfunc_tetra")
_Vfunc_hex = mod.get_function("Vfunc_hex")

# Mfuncs and include_bc (which uses _Mfuncs)
# DIPLOIDS
_Mfunc2D = mod.get_function("Mfunc2D")
_Mfunc3D = mod.get_function("Mfunc3D")
_Mfunc4D = mod.get_function("Mfunc4D")
_Mfunc5D = mod.get_function("Mfunc5D")
_include_bc = mod.get_function("include_bc")

# AUTOTETRAPLOIDS
_Mfunc2D_auto = mod.get_function("Mfunc2D_auto")
_Mfunc3D_auto = mod.get_function("Mfunc3D_auto")
_Mfunc4D_auto = mod.get_function("Mfunc4D_auto")
_Mfunc5D_auto = mod.get_function("Mfunc5D_auto")
_include_bc_auto = mod.get_function("include_bc_auto")

# ALLOTETRAPLOIDS - Subgenome A
_Mfunc2D_allo_a = mod.get_function("Mfunc2D_allo_a")
_Mfunc3D_allo_a = mod.get_function("Mfunc3D_allo_a")
_Mfunc4D_allo_a = mod.get_function("Mfunc4D_allo_a")
_Mfunc5D_allo_a = mod.get_function("Mfunc5D_allo_a")
_include_bc_allo_a = mod.get_function("include_bc_allo_a")

# ALLOTETRAPLOIDS - Subgenome B
_Mfunc2D_allo_b = mod.get_function("Mfunc2D_allo_b")
_Mfunc3D_allo_b = mod.get_function("Mfunc3D_allo_b")
_Mfunc4D_allo_b = mod.get_function("Mfunc4D_allo_b")
_Mfunc5D_allo_b = mod.get_function("Mfunc5D_allo_b")
_include_bc_allo_b = mod.get_function("include_bc_allo_b")

# AUTOHEXAPLOIDS
_Mfunc2D_autohex = mod.get_function("Mfunc2D_autohex")
_Mfunc3D_autohex = mod.get_function("Mfunc3D_autohex")
_Mfunc4D_autohex = mod.get_function("Mfunc4D_autohex")
_Mfunc5D_autohex = mod.get_function("Mfunc5D_autohex")
_include_bc_autohex = mod.get_function("include_bc_autohex")

# ALLOAUTOHEXAPLOIDS - tetraploid subgenome
_Mfunc2D_hex_tetra = mod.get_function("Mfunc2D_hex_tetra")
_Mfunc3D_hex_tetra = mod.get_function("Mfunc3D_hex_tetra")
_Mfunc4D_hex_tetra = mod.get_function("Mfunc4D_hex_tetra")
_Mfunc5D_hex_tetra = mod.get_function("Mfunc5D_hex_tetra")
_include_bc_hex_tetra = mod.get_function("include_bc_hex_tetra")

# ALLOAUTOHEXAPLOIDS - diploid subgenome
_Mfunc2D_hex_dip = mod.get_function("Mfunc2D_hex_dip")
_Mfunc3D_hex_dip = mod.get_function("Mfunc3D_hex_dip")
_Mfunc4D_hex_dip = mod.get_function("Mfunc4D_hex_dip")
_Mfunc5D_hex_dip = mod.get_function("Mfunc5D_hex_dip")
_include_bc_hex_dip = mod.get_function("include_bc_hex_dip")

# ALLOALLOHEXAPLOIDS - a subgenome
_Mfunc3D_hex_a = mod.get_function("Mfunc3D_hex_a")
_Mfunc4D_hex_a = mod.get_function("Mfunc4D_hex_a")
_Mfunc5D_hex_a = mod.get_function("Mfunc5D_hex_a")
_include_bc_hex_a = mod.get_function("include_bc_hex_a")

# ALLOALLOHEXAPLOIDS - b subgenome
_Mfunc3D_hex_b = mod.get_function("Mfunc3D_hex_b")
_Mfunc4D_hex_b = mod.get_function("Mfunc4D_hex_b")
_Mfunc5D_hex_b = mod.get_function("Mfunc5D_hex_b")
_include_bc_hex_b = mod.get_function("include_bc_hex_b")

# ALLOALLOHEXAPLOIDS - c subgenome
_Mfunc3D_hex_c = mod.get_function("Mfunc3D_hex_c")
_Mfunc4D_hex_c = mod.get_function("Mfunc4D_hex_c")
_Mfunc5D_hex_c = mod.get_function("Mfunc5D_hex_c")
_include_bc_hex_c = mod.get_function("include_bc_hex_c")    
