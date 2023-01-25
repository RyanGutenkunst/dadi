from subprocess import getstatusoutput
import dadi

def test_1d_growth():
    # Tests whether code runs
    fs = dadi.Demographics1D.growth([2,0.1], (17,), 60)