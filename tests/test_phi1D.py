import unittest
import numpy, scipy
import dadi

def phi_1D_Huber(xx, nu=1.0, theta0=1.0, gamma=0, h=0.5, theta=None, beta=1):
    """
    Version from Christian Huber that uses arbitrary precision arithmetic.
    """
    import mpmath
    mpmath.mp.dps = 5
    gamma = gamma * 4.*beta/(beta+1.)**2

    ints = numpy.empty(len(xx))*mpmath.convert(0)
    integrand = lambda xi: mpmath.exp(-4*gamma*h*xi - 2*gamma*(1-2*h)*xi**2)
    val = mpmath.quad(integrand, [0, 1])
    int0 = val
    for ii,q in enumerate(xx):
        val = mpmath.quad(integrand, [q, 1])
        ints[ii] = val
    phi = numpy.exp(4*gamma*h*xx + 2*gamma*(1-2*h)*xx**2)*ints/int0

    phi[1:-1] *= 1./(xx[1:-1]*(1-xx[1:-1]))
    phi[0] = phi[1]
    phi[-1] = 1./int0

    return phi * nu*theta0 * 4.*beta/(beta+1.)**2

def phi_1D_old(xx, nu=1.0, theta0=1.0, gamma=0, h=0.5, theta=None, beta=1):
    """
    Old version, before fix to deal with large gammas
    """
    gamma = gamma * 4.*beta/(beta+1.)**2

    ints = numpy.empty(len(xx))
    integrand = lambda xi: numpy.exp(-4*gamma*h*xi - 2*gamma*(1-2*h)*xi**2)
    val, eps = scipy.integrate.quad(integrand, 0, 1)
    int0 = val
    for ii,q in enumerate(xx):
        val, eps = scipy.integrate.quad(integrand, q, 1)
        ints[ii] = val
    phi = numpy.exp(4*gamma*h*xx + 2*gamma*(1-2*h)*xx**2)*ints/int0

    phi[1:-1] *= 1./(xx[1:-1]*(1-xx[1:-1]))
    phi[0] = phi[1]
    phi[-1] = 1./int0
    return phi * nu*theta0 * 4.*beta/(beta+1.)**2

def reldiff(x1,x2, eps=1e-12):
    return 2*abs(x1 - x2)/(x1+x2+eps)

class phi1DTestCase(unittest.TestCase):
    """
    Test routines for generating input phi's.

    These tests are primarily motivated by the desire to avoid divide by
    zero warnings.
    """
    def setUp(self):
        # Ensures that divide by zero raised an exception
        self.prior_seterr = numpy.seterr(divide='raise')

    def tearDown(self):
        # Restores seterr state for other tests
        numpy.seterr(**self.prior_seterr)

    xx = dadi.Numerics.default_grid(20)
    def test_snm(self):
        """
        Test standard neutral model.
        """
        dadi.PhiManip.phi_1D(self.xx)

    def test_genic(self):
        """
        Test non-dominant genic selection.
        """
        dadi.PhiManip.phi_1D(self.xx, gamma=1)
        dadi.PhiManip.phi_1D(self.xx, gamma=-500)

    def test_dominance(self):
        """
        Test selection with dominance.
        """
        dadi.PhiManip.phi_1D(self.xx, gamma=1, h=0.3)

    def test_dominance_positive_gamma_new_v_old(self):
        # Check that we didn't break positive gamma, by comparing
        # with the old version, for non-divergent values.
        for gamma,h in ([50, 0.14], [50, 0.84]):
            phi_orig = phi_1D_old(self.xx, gamma=gamma, h=h)
            phi_new = dadi.PhiManip.phi_1D(self.xx, gamma=gamma, h=h)
            assert(numpy.allclose(phi_orig, phi_new))

    def test_dominance_positive_gamma_divergence(self):
        # Check that divergence is fixed for positive gamma.
        for gamma,h in ([500, 0.14], [500, 0.84]):
            phi_orig = phi_1D_old(self.xx, gamma=gamma, h=h)
            phi_new = dadi.PhiManip.phi_1D(self.xx, gamma=gamma, h=h)
            assert(numpy.any(numpy.isnan(phi_orig)))
            assert(~numpy.any(numpy.isnan(phi_new)))

    def test_negative_gamma_divergence(self):
        try:
            import mpmath
        except ImportError:
            # Don't run test with we don't have mpmath library installed
            return

        # For negative gamma, compare our implementation with Christian Huber's
        # arbitrary precision arithmetic version
        for gamma, h in [(-500, 0.14), (-500, 0.94), (-2000, 0.34),
                         (-1429, 0.51), (-1e4, 0), (-1e5, 0), (-1e6, 0)]:
            phi_new = dadi.PhiManip.phi_1D(self.xx, gamma=gamma, h=h)
            phi_H = phi_1D_Huber(self.xx, gamma=gamma, h=h)
            assert(max(reldiff(phi_new, phi_H)) < 5e-3)

suite = unittest.TestLoader().loadTestsFromTestCase(phi1DTestCase)

if __name__ == '__main__':
    unittest.main()
