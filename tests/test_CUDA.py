import unittest
import dadi
import numpy as np

class CUDATestCase(unittest.TestCase):
    def test_tranpose_gpuarray(self):
        import dadi.cuda
        import pycuda.gpuarray as gpuarray

        a = np.random.uniform(size=(2, 3))
        a_gpu = gpuarray.to_gpu(a)
        aT_gpu = gpuarray.empty(a.shape[::-1], dtype=a_gpu.dtype)
        dadi.cuda.transpose_gpuarray(a_gpu, aT_gpu)

        self.assertTrue(np.allclose(aT_gpu.get(), a.transpose()))

    def test_2d_const_params(self):
        """
        """
        args = [0.1,0.1,2,0.3,0.5,-0.2,2,0.1,0.9,2]

        pts = 10
        xx = np.linspace(0, 1, pts)
        phi = np.random.uniform(size=((pts, pts)))

        dadi.enable_cuda(False)
        phi_cpu = dadi.Integration._two_pops_const_params(phi.copy(), xx, *args)
    
        dadi.enable_cuda()
        phi_gpu = dadi.Integration._two_pops_const_params(phi.copy(), xx, *args)

        self.assertTrue(np.allclose(phi_cpu, phi_gpu))

suite=unittest.TestLoader().loadTestsFromTestCase(CUDATestCase)

if __name__ == '__main__':
    try:
        dadi.enable_cuda()
        unittest.main()
    except ImportError:
        print("Failed to load dadi CUDA module")
