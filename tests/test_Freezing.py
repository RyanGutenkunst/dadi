import os
import unittest
import numpy
import dadi
from dadi.Integration import one_pop, two_pops, three_pops

def rel_abs_diff(a,b):
    return abs(a-b)/((a+b)/2)

class FreezingTestCase(unittest.TestCase):
    def test_1d_freeze(self):
        """
        Simple test of 1d freezing.
        """
        # This case is easy, phi should just not change.
        xx = dadi.Numerics.default_grid(30)
        phi_orig = dadi.PhiManip.phi_1D(xx)
        phi_frozen = one_pop(phi_orig, xx, T=0.1, nu=0.1, gamma=0.2, 
                             frozen=True)
        self.assert_(numpy.allclose(phi_frozen, phi_orig, rtol=1e-6))

    def test_2d_freeze_marginal(self):
        """
        Test marginal spectra from 2d freezing
        """
        xx = dadi.Numerics.default_grid(30)
        phi = dadi.PhiManip.phi_1D(xx)
        phi_orig = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi_frozen1 = two_pops(phi_orig, xx, T=0.1, nu2=0.3, gamma2=0.5,
                               frozen1=True)
        phi_frozen2 = two_pops(phi_orig, xx, T=0.1, nu1=0.2, gamma1=0.3, 
                               frozen2=True)
        phi_decouple = two_pops(phi_orig, xx, T=0.1, nu1=0.2, gamma1=0.3, 
                                nu2=0.3, gamma2=0.5)
        
        phi_orig_just1 = dadi.PhiManip.remove_pop(phi_orig, xx, 2)
        phi_orig_just2 = dadi.PhiManip.remove_pop(phi_orig, xx, 1)
        phi_frozen1_just1 = dadi.PhiManip.remove_pop(phi_frozen1, xx, 2)
        phi_frozen1_just2 = dadi.PhiManip.remove_pop(phi_frozen1, xx, 1)
        phi_frozen2_just1 = dadi.PhiManip.remove_pop(phi_frozen2, xx, 2)
        phi_frozen2_just2 = dadi.PhiManip.remove_pop(phi_frozen2, xx, 1)
        phi_decouple_just1 = dadi.PhiManip.remove_pop(phi_decouple, xx, 2)
        phi_decouple_just2 = dadi.PhiManip.remove_pop(phi_decouple, xx, 1)

        # Checks that if I freeze a population, the marginal spectrum
        # for that population does not change during integration.
        self.assert_(numpy.allclose(phi_orig_just1[1:-1], 
                                    phi_frozen1_just1[1:-1], 
                                    rtol=1e-4))
        self.assert_(numpy.allclose(phi_orig_just2[1:-1], 
                                    phi_frozen2_just2[1:-1], 
                                    rtol=1e-4))

        # Checks that if I freeze a population, the other population
        # continues to evolve as if decoupled from frozen population.
        # These tests are much less precise than those above, becaue
        # my integration scheme doesn't explicitly decouple the populations.
        self.assert_(numpy.allclose(phi_frozen2_just1[1:-1], 
                                    phi_decouple_just1[1:-1], 
                                    rtol=5e-2))
        self.assert_(numpy.allclose(phi_frozen1_just2[1:-1], 
                                    phi_decouple_just2[1:-1], 
                                    rtol=5e-2))

    def test_3d_freeze_marginal(self):
        """
        Test marginal spectra from 3d freezing
        """
        xx = dadi.Numerics.default_grid(30)
        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi_orig = dadi.PhiManip.phi_2D_to_3D_split_2(xx, phi)
        phi_frozen1 = three_pops(phi_orig, xx, T=0.3, 
                                 nu2=0.3, gamma2=0.5,
                                 nu3=0.4, gamma3=0.9,
                                 m23=0.3, m32=1.9,
                                 frozen1=True)
        phi_frozen2 = three_pops(phi_orig, xx, T=0.3, 
                                 nu1=0.2, gamma1=0.3, 
                                 nu3=0.4, gamma3=0.9,
                                 m13=1.0, m31=0.9,
                                 frozen2=True)
        phi_frozen3 = three_pops(phi_orig, xx, T=0.3, 
                                 nu1=0.2, gamma1=0.3, 
                                 nu2=0.3, gamma2=0.5,
                                 m12=0.3, m21=0.9,
                                 frozen3=True)
        phi_decouple1 = three_pops(phi_orig, xx, T=0.3, 
                                   nu1=0.2, gamma1=0.3, 
                                   nu2=0.3, gamma2=0.5,
                                   nu3=0.4, gamma3=0.9,
                                   m23=0.3, m32=1.9)
        phi_decouple2 = three_pops(phi_orig, xx, T=0.3, 
                                   nu1=0.2, gamma1=0.3, 
                                   nu2=0.3, gamma2=0.5,
                                   nu3=0.4, gamma3=0.9,
                                   m13=1.0, m31=0.9)
        phi_decouple3 = three_pops(phi_orig, xx, T=0.3, 
                                   nu1=0.2, gamma1=0.3, 
                                   nu2=0.3, gamma2=0.5,
                                   nu3=0.4, gamma3=0.9,
                                   m12=0.3, m21=0.9)

        fs_orig = dadi.Spectrum.from_phi(phi_orig, [10,10,10], 
                                         [xx,xx,xx])
        fs_frozen1 = dadi.Spectrum.from_phi(phi_frozen1, [10,10,10], 
                                            [xx,xx,xx])
        fs_decouple1 = dadi.Spectrum.from_phi(phi_decouple1, [10,10,10], 
                                              [xx,xx,xx])
        fs_frozen2 = dadi.Spectrum.from_phi(phi_frozen2, [10,10,10], 
                                            [xx,xx,xx])
        fs_decouple2 = dadi.Spectrum.from_phi(phi_decouple2, [10,10,10], 
                                              [xx,xx,xx])
        fs_frozen3 = dadi.Spectrum.from_phi(phi_frozen3, [10,10,10], 
                                            [xx,xx,xx])
        fs_decouple3 = dadi.Spectrum.from_phi(phi_decouple3, [10,10,10], 
                                              [xx,xx,xx])

        self.assert_(numpy.ma.allclose(fs_orig.marginalize([1,2]),
                              fs_frozen1.marginalize([1,2]), 
                              rtol=1e-2))
        self.assert_(numpy.ma.allclose(fs_orig.marginalize([0,2]),
                              fs_frozen2.marginalize([0,2]), 
                              rtol=1e-2))
        self.assert_(numpy.ma.allclose(fs_orig.marginalize([0,1]),
                              fs_frozen3.marginalize([0,1]), 
                              rtol=1e-2))

        self.assert_(numpy.ma.allclose(fs_frozen1.marginalize([0]),
                              fs_decouple1.marginalize([0]), 
                              rtol=1e-1))
        self.assert_(numpy.ma.allclose(fs_frozen2.marginalize([1]),
                              fs_decouple2.marginalize([1]), 
                              rtol=1e-1))
        self.assert_(numpy.ma.allclose(fs_frozen3.marginalize([2]),
                              fs_decouple3.marginalize([2]), 
                              rtol=1e-1))

    def test_2d_correctness(self):
        """
        Test marginal spectra from 3d freezing
        """
        xx = dadi.Numerics.default_grid(30)
        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = two_pops(phi, xx, T=0.1, nu2=0.3, gamma2=0.5, frozen1=True)
        fs = dadi.Spectrum.from_phi(phi, [10,10], [xx,xx])

        fs_stored = dadi.Spectrum.from_file('test_2D_frozen1.fs')
        self.assert_(numpy.ma.allclose(fs, fs_stored, rtol=1e-2))

suite = unittest.TestLoader().loadTestsFromTestCase(FreezingTestCase)

if __name__ == '__main__':
    unittest.main()
