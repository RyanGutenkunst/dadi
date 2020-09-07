import os
import unittest
import numpy
import dadi
from dadi.Integration import one_pop, two_pops, three_pops
import nlopt

class OptimizationTestCase(unittest.TestCase):
    def test_optimize_log(self):
        ns = (20,)
        func_ex = dadi.Numerics.make_extrap_log_func(dadi.Demographics1D.two_epoch)
        params = [0.5, 0.1]
        pts_l = [40,50,60]

        data = (1000*func_ex(params, ns, pts_l)).sample()

        popt, llopt = dadi.Inference.opt([0.35,0.15], data, func_ex, pts_l, 
                                         lower_bound=[0.1, 0],
                                         upper_bound=[1.0, 0.3],
                                         log_opt=True, maxtime=3)

    def test_optimize(self):
        ns = (20,)
        func_ex = dadi.Numerics.make_extrap_log_func(dadi.Demographics1D.two_epoch)
        params = [0.5, 0.1]
        pts_l = [40,50,60]

        data = (1000*func_ex(params, ns, pts_l)).sample()

        popt, llopt = dadi.Inference.opt([0.35,0.15], data, func_ex, pts_l, 
                                         lower_bound=[0.1, 0],
                                         upper_bound=[1.0, 0.3],
                                         maxtime=3)

    def test_eq_constraint(self):
        ns = (20,)
        func_ex = dadi.Numerics.make_extrap_log_func(dadi.Demographics1D.two_epoch)
        params = [0.5, 0.1]
        pts_l = [40,50,60]

        data = (1000*func_ex(params, ns, pts_l)).sample()
        def eq_cons(p,grad):
            return 0.5 - (p[0] + p[1])

        popt, llopt = dadi.Inference.opt([0.35,0.15], data, func_ex, pts_l, 
                                         lower_bound=[0.1, 0], upper_bound=[1.0, 0.3],
                                         algorithm=nlopt.LN_COBYLA,
                                         eq_constraints=[(eq_cons,1e-6)],
                                         maxtime=10)
        self.assertLess(abs(0.5-(popt[0]+popt[1])), 1e-4)

    def test_ineq_constraint(self):
        ns = (20,)
        func_ex = dadi.Numerics.make_extrap_log_func(dadi.Demographics1D.two_epoch)
        params = [0.5, 0.1]
        pts_l = [40,50,60]

        data = (1000*func_ex(params, ns, pts_l)).sample()
        def ineq_cons(p,grad):
            return (p[0] + p[1]) - 0.5

        popt, llopt = dadi.Inference.opt([0.15,0.20], data, func_ex, pts_l, 
                                         lower_bound=[1e-3, 0], upper_bound=[1.0, 0.7],
                                         algorithm=nlopt.LN_COBYLA,
                                         ineq_constraints=[(ineq_cons,1e-6)],
                                         maxtime=10)
        self.assertLess(popt[0]+popt[1], 0.5+1e-6)

suite = unittest.TestLoader().loadTestsFromTestCase(OptimizationTestCase)

if __name__ == '__main__':
    unittest.main()