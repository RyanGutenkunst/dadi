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

        dadi.cuda_enabled(False)
        phi_cpu = dadi.Integration.two_pops(phi.copy(), xx, *args)
    
        dadi.cuda_enabled(True)
        phi_gpu = dadi.Integration.two_pops(phi.copy(), xx, *args,
                enable_cuda_cached=True)

        self.assertTrue(np.allclose(phi_cpu, phi_gpu))

    def test_2d_temporal_params(self):
        nu1 = lambda t: 0.1+10*t
        m21 = lambda t: 3-20*t
        args = [0.1,nu1,2,0.3,m21,-0.2,2,0.1,0.9,2]

        pts = 20
        xx = np.linspace(0, 1, pts)
        phi = np.random.uniform(size=((pts, pts)))

        dadi.cuda_enabled(False)
        phi_cpu = dadi.Integration.two_pops(phi.copy(), xx, *args)
    
        dadi.cuda_enabled(True)
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

        dadi.cuda_enabled(False)
        phi_cpu = dadi.Integration.three_pops(phi, xx, T, nu1, nu2, nu3,
                       m12, m13, m21, m23, m31, m32,
                       gamma1, gamma2, gamma3, h1, h2, h3,
                       theta0, initial_t, frozen1, frozen2,
                       frozen3)

        dadi.cuda_enabled(True)
        phi_gpu = dadi.Integration.three_pops(phi, xx, T, nu1, nu2, nu3,
                       m12, m13, m21, m23, m31, m32,
                       gamma1, gamma2, gamma3, h1, h2, h3,
                       theta0, initial_t, frozen1, frozen2,
                       frozen3, enable_cuda_cached=True)
        
        self.assertTrue(np.allclose(phi_cpu, phi_gpu))

        # Need to handle frozen populations carefully in the function,
        # so we test all cases here.
        m12, m13, m21, m23, m31, m32 = [0]*6
        for frozen1 in [True, False]:
            for frozen2 in [True, False]:
                for frozen3 in [True, False]:
                    dadi.cuda_enabled(False)
                    phi_cpu = dadi.Integration.three_pops(phi, xx, T, nu1, nu2, nu3,
                                   m12, m13, m21, m23, m31, m32,
                                   gamma1, gamma2, gamma3, h1, h2, h3,
                                   theta0, initial_t, frozen1, frozen2,
                                   frozen3)

                    dadi.cuda_enabled(True)
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

        dadi.cuda_enabled(False)
        phi_cpu = dadi.Integration.three_pops(phi, xx, T, nu1, nu2, nu3,
                       m12, m13, m21, m23, m31, m32,
                       gamma1, gamma2, gamma3, h1, h2, h3,
                       theta0, initial_t, frozen1, frozen2,
                       frozen3)

        dadi.cuda_enabled(True)
        phi_gpu = dadi.Integration.three_pops(phi, xx, T, nu1, nu2, nu3,
                       m12, m13, m21, m23, m31, m32,
                       gamma1, gamma2, gamma3, h1, h2, h3,
                       theta0, initial_t, frozen1, frozen2,
                       frozen3)
        
        self.assertTrue(np.allclose(phi_cpu, phi_gpu))

        m12, m13, m21, m23, m31, m32 = [0]*6
        for frozen1 in [True, False]:
            for frozen2 in [True, False]:
                for frozen3 in [True, False]:
                    dadi.cuda_enabled(False)
                    phi_cpu = dadi.Integration.three_pops(phi, xx, T, nu1, nu2, nu3,
                                   m12, m13, m21, m23, m31, m32,
                                   gamma1, gamma2, gamma3, h1, h2, h3,
                                   theta0, initial_t, frozen1, frozen2,
                                   frozen3)

                    dadi.cuda_enabled(True)
                    phi_gpu = dadi.Integration.three_pops(phi, xx, T, nu1, nu2, nu3,
                                   m12, m13, m21, m23, m31, m32,
                                   gamma1, gamma2, gamma3, h1, h2, h3,
                                   theta0, initial_t, frozen1, frozen2,
                                   frozen3)
        
                    self.assertTrue(np.allclose(phi_cpu, phi_gpu))
                    
    def test_4d_integration(self):
        pts = 10
        nu1 = lambda t: 0.5 + 5*t
        nu2 = lambda t: 10-20*t
        nu3, nu4 = 0.3, 0.9
        m12, m13, m14 = 2.0, 0.1, 3.2
        m21, m23, m24 = lambda t: 0.5+3*t, 0.2, 1.2
        m31, m32, m34 = 0.9, 1.7, 0.9
        m41, m42, m43 = 0.3, 0.4, 1.9
        gamma1 = lambda t: -2*t
        gamma2, gamma3, gamma4 = 3.0, -1, 0.5
        h1 = lambda t: 0.2+t
        h2 = lambda t: 0.9-t
        h3, h4 = 0.3, 0.5
        theta0 = lambda t: 1 + 2*t
        f1,f2,f3,f4 = False, False, False, False
        T = 0.1

        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
        phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
        
        dadi.cuda_enabled(True)
        phi_gpu = dadi.Integration.four_pops(phi.copy(), xx, T=T, nu1=nu1, nu2=nu2, nu3=nu3, nu4=nu4,
                                             m12=m12, m13=m13, m14=m14, m21=m21, m23=m23, m24=m24,
                                             m31=m31, m32=m32, m34=m34, m41=m41, m42=m42, m43=m43,
                                             gamma1=gamma1, gamma2=gamma2, gamma3=gamma3, gamma4=gamma4,
                                             h1=h1, h2=h2, h3=h3, h4=h4, theta0=theta0,
                                             frozen1=f1, frozen2=f2, frozen3=f3, frozen4=f4)
        dadi.cuda_enabled(False)
        phi_cpu = dadi.Integration.four_pops(phi.copy(), xx, T=T, nu1=nu1, nu2=nu2, nu3=nu3, nu4=nu4,
                                             m12=m12, m13=m13, m14=m14, m21=m21, m23=m23, m24=m24,
                                             m31=m31, m32=m32, m34=m34, m41=m41, m42=m42, m43=m43,
                                             gamma1=gamma1, gamma2=gamma2, gamma3=gamma3, gamma4=gamma4,
                                             h1=h1, h2=h2, h3=h3, h4=h4, theta0=theta0,
                                             frozen1=f1, frozen2=f2, frozen3=f3, frozen4=f4)

        self.assertTrue(np.allclose(phi_cpu, phi_gpu))

        m12, m13, m14, m21, m23, m24, m31, m32, m34, m41, m42, m43 = [0]*12
        for f1 in [True, False]:
            for f2 in [True, False]:
                for f3 in [True, False]:
                    for f4 in [True, False]:
                        dadi.cuda_enabled(True)
                        phi_gpu = dadi.Integration.four_pops(phi.copy(), xx, T=T, nu1=nu1, nu2=nu2, nu3=nu3, nu4=nu4,
                                                             m12=m12, m13=m13, m14=m14, m21=m21, m23=m23, m24=m24,
                                                             m31=m31, m32=m32, m34=m34, m41=m41, m42=m42, m43=m43,
                                                             gamma1=gamma1, gamma2=gamma2, gamma3=gamma3, gamma4=gamma4,
                                                             h1=h1, h2=h2, h3=h3, h4=h4, theta0=theta0,
                                                             frozen1=f1, frozen2=f2, frozen3=f3, frozen4=f4)
                        dadi.cuda_enabled(False)
                        phi_cpu = dadi.Integration.four_pops(phi.copy(), xx, T=T, nu1=nu1, nu2=nu2, nu3=nu3, nu4=nu4,
                                                             m12=m12, m13=m13, m14=m14, m21=m21, m23=m23, m24=m24,
                                                             m31=m31, m32=m32, m34=m34, m41=m41, m42=m42, m43=m43,
                                                             gamma1=gamma1, gamma2=gamma2, gamma3=gamma3, gamma4=gamma4,
                                                             h1=h1, h2=h2, h3=h3, h4=h4, theta0=theta0,
                                                             frozen1=f1, frozen2=f2, frozen3=f3, frozen4=f4)
                        self.assertTrue(np.allclose(phi_cpu, phi_gpu))

    def test_5d_integration(self):
        kwargs = {'T': 0.1,
                  'nu1': 0.2, 'nu2': 1.3, 'nu3': 7.1, 'nu4': 27.1, 'nu5':lambda t: 0.9-8*t,
                  'm12': 3, 'm13': 2.9, 'm14': 0.9, 'm15': 10,
                  'm21': 3, 'm23': lambda t: 0.9+10*t, 'm24': 0.9, 'm25': lambda t: 2-15*t,
                  'm31': 3.5, 'm32': 2.9, 'm34': 0.9, 'm35': 10,
                  'm41': 3.3, 'm42': 2.2, 'm43': 0.1, 'm45': 9,
                  'm51': 3.3, 'm52': 2.2, 'm53': 0.8, 'm54': 9.2,
                  'gamma1': -1, 'gamma2': 2, 'gamma3': -1.9, 'gamma4': lambda t: 10*t, 'gamma5': -1, 
                  'h1': 0.1, 'h2': 0.9, 'h3': 0.2, 'h4': 0.9, 'h5': 0.1, 
                  'frozen1':False, 'frozen2':False, 'frozen3':False, 'frozen4':False, 'frozen5':False,
                  'theta0': lambda t: 10*t}

        pts = 5
        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
        phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
        phi = dadi.PhiManip.phi_4D_to_5D(phi, 0,0,0, xx,xx,xx,xx,xx)
        dadi.cuda_enabled(True)
        phi_gpu = dadi.Integration.five_pops(phi.copy(), xx, **kwargs)
        dadi.cuda_enabled(False)
        phi_cpu = dadi.Integration.five_pops(phi.copy(), xx, **kwargs)

        self.assertTrue(np.allclose(phi_cpu, phi_gpu))

if dadi.cuda_enabled(True):
    suite = unittest.TestLoader().loadTestsFromTestCase(CUDATestCase)
else:
    suite = unittest.TestSuite()

if __name__ == '__main__':
    if dadi.cuda_enabled(True):
        unittest.main()
