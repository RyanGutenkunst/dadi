import unittest
import numpy as np
import dadi.tridiag_cython as tridiag

class TridiagonalTestCase(unittest.TestCase):
    """
    Test tridiagonal solving routines
    """
    # Generate a random tridiagonal test case
    n = 100

    a = np.random.rand(n)
    a[0] = 0
    b = np.random.rand(n)
    c = np.random.rand(n)
    c[-1] = 0
    r = np.random.rand(n)
    
    # Create the corresponding array
    arr = np.zeros((n,n))
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
        u = tridiag.tridiag(self.a,self.b,self.c,self.r)
        rcheck = np.dot(self.arr,u)

        self.assertTrue(np.allclose(self.r, rcheck, atol=1e-8))

    def test_tridiag_single(self):
        """
        Test single precision tridiagonal routine
        """
        # Now that we're using Cython, we don't get the automatic type
        # conversion that f2py gave us. But type-checking on every
        # function call is slow.
        u = tridiag.tridiag_fl(np.asarray(self.a, dtype=np.float32),
                               np.asarray(self.b, dtype=np.float32),
                               np.asarray(self.c, dtype=np.float32), 
                               np.asarray(self.r, dtype=np.float32))
        rcheck = np.dot(self.arr,u)

        self.assertTrue(np.allclose(self.r, rcheck, atol=1e-2))

suite = unittest.TestLoader().loadTestsFromTestCase(TridiagonalTestCase)

if __name__ == '__main__':
    unittest.main()
