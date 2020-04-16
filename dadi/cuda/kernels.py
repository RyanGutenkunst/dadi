from pycuda.compiler import SourceModule

mod = SourceModule("""
  __global__ void inject_mutations_2D(double *phi, int M, double val01, double val10)
  {
    phi[1] += val01;
    phi[M] += val10;
  }
  """)
_inject_mutations_2D_vals = mod.get_function("inject_mutations_2D")