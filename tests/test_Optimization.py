import os
import unittest
import numpy
import dadi
from dadi.Integration import one_pop, two_pops, three_pops

class OptimizationTestCase(unittest.TestCase):
    def test_optimize_log(self):
        ns = (20,)
        func_ex = dadi.Numerics.make_extrap_log_func(dadi.Demographics1D.two_epoch)
        params = [0.5, 0.1]
        pts_l = [40,50,60]

        data = (1000*func_ex(params, ns, pts_l)).sample()

        result = dadi.Inference.optimize_log([0.35,0.15], data, func_ex, pts_l, 
                                             lower_bound=[0.1, 0], 
                                             upper_bound=[1.0, 0.3],
                                             maxiter=3)
        
    def test_optimize_file_output(self):
        ns = (20,)
        func_ex = dadi.Numerics.make_extrap_log_func(dadi.Demographics1D.two_epoch)
        params = [0.5, 0.1]
        pts_l = [40,50,60]

        data = (1000*func_ex(params, ns, pts_l)).sample()

        result = dadi.Inference.optimize_log([0.35,0.15], data, func_ex, pts_l, 
                                             lower_bound=[0.1, 0], 
                                             upper_bound=[1.0, 0.3],
                                             maxiter=3,
                                             verbose=1, output_file='test.out')
        self.assertTrue(os.path.exists('test.out'))
        os.remove('test.out')


suite = unittest.TestLoader().loadTestsFromTestCase(OptimizationTestCase)

if __name__ == '__main__':
    unittest.main()
