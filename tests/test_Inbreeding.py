import unittest
import dadi

class InbreedingTestCase(unittest.TestCase):
    def test_inbreeding_sfs(self):
        # Test the creation of spectra with inbreeding from 1D, 2D, 
        # and 3D DAFs.
        xx = dadi.Numerics.default_grid(50)
        phi = dadi.PhiManip.phi_1D(xx)
        sfs_1d = dadi.Spectrum.from_phi_inbreeding(phi, (20,), (xx,), (0.5,), (2,))
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        sfs_2d = dadi.Spectrum.from_phi_inbreeding(phi, (20,20), (xx,xx), (0.5,0.5), (2,2))
        phi = dadi.PhiManip.phi_2D_to_3D_split_1(xx, phi)
        sfs_3d = dadi.Spectrum.from_phi_inbreeding(phi, (20,20,20), (xx,xx,xx), (0.5,0.5,0.5), (2,2,2))

suite = unittest.TestLoader().loadTestsFromTestCase(InbreedingTestCase)

if __name__ == '__main__':
    unittest.main()
