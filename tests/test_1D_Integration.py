import unittest
import dadi

class ResultsTestCase(unittest.TestCase):
    def test_1d_growth(self):
        # Tests whether code runs
        fs = dadi.Demographics1D.growth([2,0.1], (17,), 60)

suite = unittest.TestLoader().loadTestsFromTestCase(ResultsTestCase)

if __name__ == '__main__':
    unittest.main()
