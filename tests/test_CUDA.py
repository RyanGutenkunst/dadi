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
        args = [0.1,0.1,2,0.3,0.5,-0.2,2,0.1,0.9,2]

        pts = 20
        xx = np.linspace(0, 1, pts)
        phi = np.random.uniform(size=((pts, pts)))

        dadi.enable_cuda(False)
        phi_cpu = dadi.Integration.two_pops(phi.copy(), xx, *args)
    
        dadi.enable_cuda()
        phi_gpu = dadi.Integration.two_pops(phi.copy(), xx, *args)

        self.assertTrue(np.allclose(phi_cpu, phi_gpu))

    def test_2d_temporal_params(self):
        nu1 = lambda t: 0.1+10*t
        m21 = lambda t: 3-20*t
        args = [0.1,nu1,2,0.3,m21,-0.2,2,0.1,0.9,2]

        pts = 20
        xx = np.linspace(0, 1, pts)
        phi = np.random.uniform(size=((pts, pts)))

        dadi.enable_cuda(False)
        phi_cpu = dadi.Integration.two_pops(phi.copy(), xx, *args)
    
        dadi.enable_cuda()
        phi_gpu = dadi.Integration.two_pops(phi.copy(), xx, *args)

        self.assertTrue(np.allclose(phi_cpu, phi_gpu))

    def test_3d_const_params(self):
        pts = 13

        nu1, nu2, nu3 = [2,1,0.1]
        m12, m13, m21, m23, m31, m32 = [0.1,3,10,0,0.3,0.1]
        gamma1, gamma2, gamma3 = [-1,2.0,0.1]
        h1, h2, h3 = [0.2,0.3,0.9]
        theta0, initial_t, T = [10.2, 0.1, 0.1+0.1]
        frozen1, frozen2, frozen3 = False, False, False

        xx = np.linspace(0,1,pts)
        np.random.seed(213)
        phi = np.random.uniform(size=(pts,pts,pts))

        dadi.enable_cuda(False)
        phi_cpu = dadi.Integration.three_pops(phi, xx, T, nu1, nu2, nu3,
                       m12, m13, m21, m23, m31, m32,
                       gamma1, gamma2, gamma3, h1, h2, h3,
                       theta0, initial_t, frozen1, frozen2,
                       frozen3)

        dadi.enable_cuda()
        phi_gpu = dadi.Integration.three_pops(phi, xx, T, nu1, nu2, nu3,
                       m12, m13, m21, m23, m31, m32,
                       gamma1, gamma2, gamma3, h1, h2, h3,
                       theta0, initial_t, frozen1, frozen2,
                       frozen3)
        
        self.assertTrue(np.allclose(phi_cpu, phi_gpu))

        # Need to handle frozen populations carefully in the function,
        # so we test all cases here.
        m12, m13, m21, m23, m31, m32 = [0]*6
        for frozen1 in [True, False]:
            for frozen2 in [True, False]:
                for frozen3 in [True, False]:
                    dadi.enable_cuda(False)
                    phi_cpu = dadi.Integration.three_pops(phi, xx, T, nu1, nu2, nu3,
                                   m12, m13, m21, m23, m31, m32,
                                   gamma1, gamma2, gamma3, h1, h2, h3,
                                   theta0, initial_t, frozen1, frozen2,
                                   frozen3)

                    dadi.enable_cuda()
                    phi_gpu = dadi.Integration.three_pops(phi, xx, T, nu1, nu2, nu3,
                                   m12, m13, m21, m23, m31, m32,
                                   gamma1, gamma2, gamma3, h1, h2, h3,
                                   theta0, initial_t, frozen1, frozen2,
                                   frozen3)
        
                    self.assertTrue(np.allclose(phi_cpu, phi_gpu))

    def test_3d_temporal_params(self):
        pts = 17

        nu1, nu2, nu3 = [2,1,0.1]
        nu1 = lambda t: 0.1+4*t
        m12, m13, m21, m23, m31, m32 = [0.1,3,10,0,0.3,0.1]
        m23 = lambda t: 9-10*t
        gamma1, gamma2, gamma3 = [-1,2.0,0.1]
        gamma3 = lambda t: t
        h1, h2, h3 = [0.2,0.3,0.9]
        h3 = lambda t: 0.1+2*t
        theta0, initial_t, T = [10.2, 0.1, 0.1+0.1]
        frozen1, frozen2, frozen3 = False, False, False

        xx = np.linspace(0,1,pts)
        np.random.seed(213)
        phi = np.random.uniform(size=(pts,pts,pts))

        dadi.enable_cuda(False)
        phi_cpu = dadi.Integration.three_pops(phi, xx, T, nu1, nu2, nu3,
                       m12, m13, m21, m23, m31, m32,
                       gamma1, gamma2, gamma3, h1, h2, h3,
                       theta0, initial_t, frozen1, frozen2,
                       frozen3)

        dadi.enable_cuda()
        phi_gpu = dadi.Integration.three_pops(phi, xx, T, nu1, nu2, nu3,
                       m12, m13, m21, m23, m31, m32,
                       gamma1, gamma2, gamma3, h1, h2, h3,
                       theta0, initial_t, frozen1, frozen2,
                       frozen3)
        
        self.assertTrue(np.allclose(phi_cpu, phi_gpu))

        # Need to handle frozen populations carefully in the function,
        # so we test all cases here.
        m12, m13, m21, m23, m31, m32 = [0]*6
        for frozen1 in [True, False]:
            for frozen2 in [True, False]:
                for frozen3 in [True, False]:
                    dadi.enable_cuda(False)
                    phi_cpu = dadi.Integration.three_pops(phi, xx, T, nu1, nu2, nu3,
                                   m12, m13, m21, m23, m31, m32,
                                   gamma1, gamma2, gamma3, h1, h2, h3,
                                   theta0, initial_t, frozen1, frozen2,
                                   frozen3)

                    dadi.enable_cuda()
                    phi_gpu = dadi.Integration.three_pops(phi, xx, T, nu1, nu2, nu3,
                                   m12, m13, m21, m23, m31, m32,
                                   gamma1, gamma2, gamma3, h1, h2, h3,
                                   theta0, initial_t, frozen1, frozen2,
                                   frozen3)
        
                    self.assertTrue(np.allclose(phi_cpu, phi_gpu))

suite=unittest.TestLoader().loadTestsFromTestCase(CUDATestCase)

if __name__ == '__main__':
    try:
        dadi.enable_cuda()
        unittest.main()
    except ImportError:
        print("Failed to load dadi CUDA module")
