import unittest
import numpy as np
import dadi
from dadi.Misc import combine_pops

class SubgenomesTestCase(unittest.TestCase):
    def test_combine_pops(self):
        sfs = dadi.Demographics2D.snm(None, (20,20), 50)
        sfs2 = dadi.Spectrum(combine_pops(sfs))
    
    def test_3d_combinations(self):
        xx = dadi.Numerics.default_grid(50)
        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.PhiManip.phi_2D_to_3D_split_1(xx, phi)
        sfs    = dadi.Spectrum.from_phi(phi, (20,20,20), (xx,xx,xx))
        sfs1   = dadi.Spectrum(combine_pops(sfs))
        sfs1_1 = dadi.Spectrum(combine_pops(sfs1))
        sfs2   = dadi.Spectrum(combine_pops(sfs, idx=[1,2]))
        sfs2_1 = dadi.Spectrum(combine_pops(sfs2))
        sfs3   = dadi.Spectrum(combine_pops(sfs, idx=[0,2]))
        sfs3_1 = dadi.Spectrum(combine_pops(sfs3))

suite = unittest.TestLoader().loadTestsFromTestCase(SubgenomesTestCase)

if __name__ == '__main__':
    unittest.main()
