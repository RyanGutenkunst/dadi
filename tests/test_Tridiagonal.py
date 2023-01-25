import pytest
import numpy
import dadi

try:
    import dadi.tridiag_cython
    skip = True
except:
    skip = False

@pytest.fixture
def test_details():
    """
    Tridiagonal solving routines
    """
    # Generate a random tridiagonal test case
    n = 100

    pytest.a = numpy.random.rand(n)
    pytest.a[0] = 0
    pytest.b = numpy.random.rand(n)
    pytest.c = numpy.random.rand(n)
    pytest.c[-1] = 0
    pytest.r = numpy.random.rand(n)

    # Create the corresponding array
    pytest.arr = numpy.zeros((n,n))
    for ii,row in enumerate(pytest.arr):
        if ii != 0:
            row[ii-1] = pytest.a[ii]
        row[ii] = pytest.b[ii]
        if ii != n-1:
            row[ii+1] = pytest.c[ii]


def test_tridiag_double(test_details):
    """
    Test double precision tridiagonal routine
    """
    u = dadi.tridiag_cython.tridiag(pytest.a,pytest.b,pytest.c,pytest.r)
    rcheck = numpy.dot(pytest.arr,u)

    assert(numpy.allclose(pytest.r, rcheck, atol=1e-8))

@pytest.mark.skipif(skip, reason="tridiag_fl function needs to be updated.")
def test_tridiag_single(test_details):
    """
    Test single precision tridiagonal routine
    """
    u = dadi.tridiag_cython.tridiag_fl(pytest.a,pytest.b,pytest.c,pytest.r)
    rcheck = numpy.dot(pytest.arr,u)

    assert(numpy.allclose(pytest.r, rcheck, atol=1e-3))
