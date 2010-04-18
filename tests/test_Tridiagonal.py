import unittest
import numpy
import dadi

class TridiagonalTestCase(unittest.TestCase):
    """
    Test tridiagonal solving routines
    """
    # Generate a random tridiagonal test case
    n = 100

    a = numpy.random.rand(n)
    a[0] = 0
    b = numpy.random.rand(n)
    c = numpy.random.rand(n)
    c[-1] = 0
    r = numpy.random.rand(n)
    
    # Create the corresponding array
    arr = numpy.zeros((n,n))
    for ii,row in enumerate(arr):
        if ii != 0:
            row[ii-1] = a[ii]
        row[ii] = b[ii]
        if ii != n-1:
            row[ii+1] = c[ii]

    def test_tridiag_double(self):
        """
        Test double precision tridiagonal routine
        """
        u = dadi.tridiag.tridiag(self.a,self.b,self.c,self.r)
        rcheck = numpy.dot(self.arr,u)

        self.assert_(numpy.allclose(self.r, rcheck, atol=1e-10))

    def test_tridiag_single(self):
        """
        Test single precision tridiagonal routine
        """
        u = dadi.tridiag.tridiag_fl(self.a,self.b,self.c,self.r)
        rcheck = numpy.dot(self.arr,u)

        self.assert_(numpy.allclose(self.r, rcheck, atol=1e-3))

suite = unittest.TestLoader().loadTestsFromTestCase(TridiagonalTestCase)
