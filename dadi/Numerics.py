"""
Numerically useful functions, including extrapolation and default grid.
"""
import logging
logger = logging.getLogger('Numerics')

import functools, os
import numpy
# Account for difference in scipy installations.
try:
    from scipy.misc import comb
except ImportError:
    from scipy import comb
from scipy.special import gammaln

def quadratic_grid(num_pts):
    """
    A nonuniform grid of points on [0,1] with a quadratic pattern of spacings.

    The grid is weighted to be denser near 0 and 1, which is useful for
    population genetic simulations. In between, it smoothly increases and
    then decreases the step size.
    """
    # Rounds down...
    small_pts = int(num_pts / 10)
    large_pts = num_pts - small_pts+1

    grid = numpy.linspace(0, 0.05, small_pts)

    # The interior grid spacings are a quadratic function of x, being
    # approximately x1 at the boundaries. To achieve this, our actual grid
    # positions must come from a cubic function.
    # Here I calculate and apply the appropriate conditions:
    #   x[q = 0] = x_start  =>  d = x_start
    #   dx/dq [q = 0] = dx_start => c = dx_start/dq
    #   x[q = 1] = 1
    #   dx/dq [q = 1] = dx_start
    
    q = numpy.linspace(0, 1, large_pts)
    dq = q[1] - q[0]
    x_start = grid[-1]
    dx_start = grid[-1] - grid[-2]

    d = x_start
    c = dx_start/dq
    b = -3*(-dq + dx_start + dq*x_start)/dq
    a = -2 * b/3
    grid = numpy.concatenate((grid[:-1], a*q**3 + b*q**2 + c*q + d))

    return grid

def estimate_best_exp_grid_crwd(ns):
    """
    Emperical "best" values for exponential grid crowding.

    These functional forms were estimated by running many simulations at
    different parameter values and asking when a simulation at pts_l = [max(ns),
    max(ns)+10, max(ns)+20] was most accurate.

    These cannot be considered absolute best values, as that may depend on the
    model. It does seem broadly true that the optimal value of crwd increases
    with system size, up to a point.
    """
    xx = numpy.mean(ns)
    if len(ns) == 1:
        return min(max(xx**0.4 / 1.3, 1), 9)
    elif len(ns) == 2:
        return min(max(1.5 * xx**0.4, 1), 8)
    else:
        raise ValueError('Due to computational expense, no optimum has been '
                         'determined for 3D or more models. Try sticking with '
                         'crwd=8.')

def exponential_grid(pts, crwd=8.):
    """
    An exponentially spaced grid. This is now the default grid.

    crwd controls the degree to which grid points crowd against x=0 or x=1.
    The value of crwd=8 seems to be a good default for integration with large
    systems. See estimate_best_exp_grid_crwd for some empirical optimizations
    beyond this.

    This grid was contributed by Simon Gravel.
    """
    unif = numpy.linspace(-1,1,pts)
    grid = 1./(1. + numpy.exp(-crwd*unif))

    # Normalize
    grid = (grid-grid[0])/(grid[-1]-grid[0])
    return grid

default_grid = exponential_grid

def end_point_first_derivs(xx):
    """
    Coefficients for a 5-point one-sided approximation of the first derivative.

    xx: grid on which the data to be differentiated lives

    Returns ret, a 2x5 array. ret[0] is the coefficients for an approximation
    of the derivative at xx[0]. It is used by deriv = numpy.dot(ret[0],
    yy[:5]). ret[1] is the coefficients for the derivative at xx[-1]. It can be
    used by deriv = dot(ret[1][::-1], yy[-5:]). (Note that we need to reverse
    the coefficient array here.
    """
    output = numpy.zeros((2,5))

    # These are the coefficients for a one-sided 1st derivative of f[0].
    # So f'[0] = d10_0 f[0] + d10_1 f[1] + d10_2 f[2] + d10_3 f[3] + d10_4 f[4]
    # This expression is 4th order accurate.
    # To derive it in Mathematica, use NDSolve`FiniteDifferenceDerivative[1, {xx[0], xx[1], xx[2], xx[3], xx[4]}, {f[0], f[1], f[2], f[3], f[4]}, DifferenceOrder -> 4][[1]] // Simplify
    d10_0 = (-1 + ((-1 + ((-2*xx[0] + xx[1] + xx[2]) * (-xx[0] + xx[3]))/ ((xx[0] - xx[1])*(-xx[0] + xx[2])))* (-xx[0] + xx[4]))/(-xx[0] + xx[3]))/ (-xx[0] + xx[4])
    d10_1 = -(((xx[0] - xx[2])*(xx[0] - xx[3])*(xx[0] - xx[4]))/ ((xx[0] - xx[1])*(xx[1] - xx[2])*(xx[1] - xx[3])* (xx[1] - xx[4])))
    d10_2 = ((xx[0] - xx[1])*(xx[0] - xx[3])*(xx[0] - xx[4]))/ ((xx[0] - xx[2])*(xx[1] - xx[2])*(xx[2] - xx[3])* (xx[2] - xx[4]))
    d10_3 = -(((xx[0] - xx[1])*(xx[0] - xx[2])*(xx[0] - xx[4]))/ ((xx[0] - xx[3])*(-xx[1] + xx[3])* (-xx[2] + xx[3])*(xx[3] - xx[4])))
    d10_4 = ((xx[0] - xx[1])*(xx[0] - xx[2])*(xx[0] - xx[3]))/ ((xx[0] - xx[4])*(xx[1] - xx[4])*(xx[2] - xx[4])* (xx[3] - xx[4]))

    output[0] = (d10_0, d10_1, d10_2, d10_3, d10_4)

    # These are the coefficients for a one-sided 1st derivative of f[-1].
    # So f'[-1] = d1m1_m1 f[-1] + d1m1_m2 f[-2] + d1m1_m3 f[-3] + d1m1_m4 f[-4]
    #             + d1m1_m5 f[-5]
    d1m1_m1 = (xx[-1]*((3*xx[-2] - 4*xx[-1])*xx[-1] + xx[-3]*(-2*xx[-2] + 3*xx[-1])) + xx[-4]*(xx[-3]*(xx[-2] - 2*xx[-1]) + xx[-1]*(-2*xx[-2] + 3*xx[-1])) + xx[-5]*(xx[-3]*(xx[-2] - 2*xx[-1]) + xx[-4]*(xx[-3] + xx[-2] - 2*xx[-1]) + xx[-1]*(-2*xx[-2] + 3*xx[-1])))/ ((xx[-4] - xx[-1])*(-xx[-5] + xx[-1])* (-xx[-3] + xx[-1])*(-xx[-2] + xx[-1]))
    d1m1_m2 = ((xx[-5] - xx[-1])*(xx[-4] - xx[-1])* (xx[-3] - xx[-1]))/ ((xx[-5] - xx[-2])*(-xx[-4] + xx[-2])*(-xx[-3] + xx[-2])*(xx[-2] - xx[-1]))
    d1m1_m3 = ((xx[-5] - xx[-1])*(xx[-4] - xx[-1])* (xx[-2] - xx[-1]))/ ((xx[-5] - xx[-3])*(-xx[-4] + xx[-3])* (xx[-3] - xx[-2])*(xx[-3] - xx[-1]))
    d1m1_m4 = ((xx[-5] - xx[-1])*(xx[-3] - xx[-1])* (xx[-2] - xx[-1]))/ ((xx[-5] - xx[-4])*(xx[-4] - xx[-3])* (xx[-4] - xx[-2])*(xx[-4] - xx[-1])) 
    d1m1_m5 = ((xx[-4] - xx[-1])*(xx[-3] - xx[-1])* (xx[-2] - xx[-1]))/ ((xx[-5] - xx[-4])*(xx[-5] - xx[-3])* (xx[-5] - xx[-2])*(-xx[-5] + xx[-1]))

    output[1] = (d1m1_m1, d1m1_m2, d1m1_m3, d1m1_m4, d1m1_m5)

    return output

def linear_extrap((y1, y2), (x1, x2)):
    """
    Linearly extrapolate from two x,y pairs to x = 0.

    y1,y2: y values from x,y pairs. Note that these can be arrays of values.
    x1,x2: x values from x,y pairs. These should be scalars.

    Returns extrapolated y at x=0.
    """
    return (x2 * y1 - x1 * y2)/(x2 - x1)

def quadratic_extrap((y1, y2, y3), (x1, x2, x3)):
    """
    Quadratically extrapolate from three x,y pairs to x = 0.

    y1,y2,y3: y values from x,y pairs. Note that these can be arrays of values.
    x1,x2,x3: x values from x,y pairs. These should be scalars.

    Returns extrapolated y at x=0.
    """
    return x2*x3/((x1-x2)*(x1-x3)) * y1 + x1*x3/((x2-x1)*(x2-x3)) * y2\
            + x1*x2/((x3-x1)*(x3-x2)) * y3

def cubic_extrap((y1, y2, y3, y4), (x1, x2, x3, x4)):
    """
    Cubically extrapolate from three x,y pairs to x = 0.

    y1,y2,y3: y values from x,y pairs. Note that these can be arrays of values.
    x1,x2,x3: x values from x,y pairs. These should be scalars.

    Returns extrapolated y at x=0.
    """
    # This horrid implementation came from using CForm in Mathematica.
    Power = numpy.power
    return ((x1*(x1 - x3)*x3* (x1 - x4)*(x3 - x4)*x4* y2 + x2*Power(x4,2)* (-(Power(x3,3)*y1) + Power(x3,2)*x4*y1 + Power(x1,2)* (x1 - x4)*y3) + Power(x1,2)*x2* Power(x3,2)*(-x1 + x3)* y4 + Power(x2,3)* (x4*(-(Power(x3,2)* y1) + x3*x4*y1 + x1*(x1 - x4)*y3) + x1*x3*(-x1 + x3)* y4) + Power(x2,2)* (x1*x4* (-Power(x1,2) + Power(x4,2))*y3 + Power(x3,3)* (x4*y1 - x1*y4) + x3*(-(Power(x4,3)* y1) + Power(x1,3)*y4)))/ ((x1 - x2)*(x1 - x3)* (x2 - x3)*(x1 - x4)* (x2 - x4)*(x3 - x4)))

def quartic_extrap((y1,y2,y3,y4,y5), (x1,x2,x3,x4,x5)):
    """
    Quartically extrapolate from three x,y pairs to x = 0.

    y1,y2...: y values from x,y pairs. Note that these can be arrays of values.
    x1,x2...: x values from x,y pairs. These should be scalars.

    Returns extrapolated y at x=0.
    """
    # This horrid implementation came from using CForm in Mathematica.
    Power = numpy.power
    return (-(x1*(x1 - x3)*x3*(x1 - x4)*(x3 - x4)*x4*(x1 - x5)* (x3 - x5)*(x4 - x5)*x5*y2) + Power(x2,4)*(-(x1*(x1 - x4)*x4*(x1 - x5)* (x4 - x5)*x5*y3) + Power(x3,3)*(x1*x5*(-x1 + x5)*y4 + Power(x4,2)*(x5*y1 - x1*y5) + x4*(-(Power(x5,2)*y1) + Power(x1,2)*y5)) + Power(x3,2)*(x1*x5* (Power(x1,2) - Power(x5,2))*y4 + Power(x4,3)*(-(x5*y1) + x1*y5) + x4*(Power(x5,3)*y1 - Power(x1,3)*y5)) + x3*(Power(x1,2)*Power(x5,2)*(-x1 + x5)*y4 + Power(x4,3)* (Power(x5,2)*y1 - Power(x1,2)*y5) + Power(x4,2)* (-(Power(x5,3)*y1) + Power(x1,3)*y5))) + Power(x2,3)*(x1*x4*x5* (Power(x1,3)*(x4 - x5) + x4*x5*(Power(x4,2) - Power(x5,2)) + x1*(-Power(x4,3) + Power(x5,3)))*y3 + Power(x3,4)*(x1*(x1 - x5)*x5*y4 + Power(x4,2)*(-(x5*y1) + x1*y5) + x4*(Power(x5,2)*y1 - Power(x1,2)*y5)) + x3*(Power(x1,2)*Power(x5,2)* (Power(x1,2) - Power(x5,2))*y4 + Power(x4,4)* (-(Power(x5,2)*y1) + Power(x1,2)*y5) + Power(x4,2)* (Power(x5,4)*y1 - Power(x1,4)*y5)) + Power(x3,2)*(x1*x5* (-Power(x1,3) + Power(x5,3))*y4 + Power(x4,4)*(x5*y1 - x1*y5) + x4*(-(Power(x5,4)*y1) + Power(x1,4)*y5))) + x2*(Power(x1,2)*(x1 - x4)*Power(x4,2)* (x1 - x5)*(x4 - x5)*Power(x5,2)*y3 + Power(x3,4)*(Power(x1,2)*(x1 - x5)* Power(x5,2)*y4 + Power(x4,3)* (-(Power(x5,2)*y1) + Power(x1,2)*y5) + Power(x4,2)* (Power(x5,3)*y1 - Power(x1,3)*y5)) + Power(x3,2)*(Power(x1,3)*(x1 - x5)* Power(x5,3)*y4 + Power(x4,4)* (-(Power(x5,3)*y1) + Power(x1,3)*y5) + Power(x4,3)* (Power(x5,4)*y1 - Power(x1,4)*y5)) + Power(x3,3)*(Power(x1,2)*Power(x5,2)* (-Power(x1,2) + Power(x5,2))*y4 + Power(x4,4)* (Power(x5,2)*y1 - Power(x1,2)*y5) + Power(x4,2)* (-(Power(x5,4)*y1) + Power(x1,4)*y5))) + Power(x2,2)*(x1*x4*x5* (Power(x4,2)*Power(x5,2)*(-x4 + x5) + Power(x1,3)*(-Power(x4,2) + Power(x5,2)) + Power(x1,2)*(Power(x4,3) - Power(x5,3)))*y3 + Power(x3,4)* (x1*x5*(-Power(x1,2) + Power(x5,2))*y4 + Power(x4,3)*(x5*y1 - x1*y5) + x4*(-(Power(x5,3)*y1) + Power(x1,3)*y5)) + Power(x3,3)*(x1*x5* (Power(x1,3) - Power(x5,3))*y4 + Power(x4,4)*(-(x5*y1) + x1*y5) + x4*(Power(x5,4)*y1 - Power(x1,4)*y5)) + x3*(Power(x1,3)*Power(x5,3)*(-x1 + x5)*y4 + Power(x4,4)* (Power(x5,3)*y1 - Power(x1,3)*y5) + Power(x4,3)* (-(Power(x5,4)*y1) + Power(x1,4)*y5))))/ ((x1 - x2)*(x1 - x3)*(x2 - x3)*(x1 - x4)*(x2 - x4)* (x3 - x4)*(x1 - x5)*(x2 - x5)*(x3 - x5)*(x4 - x5))

def quintic_extrap((y1,y2,y3,y4,y5,y6), (x1,x2,x3,x4,x5,x6)):
    """
    Quintic extrapolate from three x,y pairs to x = 0.

    y1,y2...: y values from x,y pairs. Note that these can be arrays of values.
    x1,x2...: x values from x,y pairs. These should be scalars.

    Returns extrapolated y at x=0.
    """
    # This horrid implementation came from using CForm in Mathematica.
    Power = numpy.power
    return (-(x1*(x1 - x3)*x3*(x1 - x4)*(x3 - x4)*x4*(x1 - x5)* (x3 - x5)*(x4 - x5)*x5*(x1 - x6)*(x3 - x6)* (x4 - x6)*(x5 - x6)*x6*y2) + Power(x2,5)*(-(x1*(x1 - x4)*x4*(x1 - x5)* (x4 - x5)*x5*(x1 - x6)*(x4 - x6)*(x5 - x6)* x6*y3) + Power(x3,4)* (-(x1*(x1 - x5)*x5*(x1 - x6)*(x5 - x6)*x6* y4) + Power(x4,3)* (x1*x6*(-x1 + x6)*y5 + Power(x5,2)*(x6*y1 - x1*y6) + x5*(-(Power(x6,2)*y1) + Power(x1,2)*y6)) + Power(x4,2)* (x1*x6*(Power(x1,2) - Power(x6,2))*y5 + Power(x5,3)*(-(x6*y1) + x1*y6) + x5*(Power(x6,3)*y1 - Power(x1,3)*y6)) + x4*(Power(x1,2)*Power(x6,2)*(-x1 + x6)* y5 + Power(x5,3)* (Power(x6,2)*y1 - Power(x1,2)*y6) + Power(x5,2)* (-(Power(x6,3)*y1) + Power(x1,3)*y6))) + Power(x3,3)* (x1*x5*x6*(Power(x1,3)*(x5 - x6) + x5*x6*(Power(x5,2) - Power(x6,2)) + x1*(-Power(x5,3) + Power(x6,3)))*y4 + Power(x4,4)* (x1*(x1 - x6)*x6*y5 + Power(x5,2)*(-(x6*y1) + x1*y6) + x5*(Power(x6,2)*y1 - Power(x1,2)*y6)) + x4*(Power(x1,2)*Power(x6,2)* (Power(x1,2) - Power(x6,2))*y5 + Power(x5,4)* (-(Power(x6,2)*y1) + Power(x1,2)*y6) + Power(x5,2)* (Power(x6,4)*y1 - Power(x1,4)*y6)) + Power(x4,2)* (x1*x6*(-Power(x1,3) + Power(x6,3))*y5 + Power(x5,4)*(x6*y1 - x1*y6) + x5*(-(Power(x6,4)*y1) + Power(x1,4)*y6))) + x3*(Power(x1,2)*(x1 - x5)*Power(x5,2)* (x1 - x6)*(x5 - x6)*Power(x6,2)*y4 + Power(x4,4)* (Power(x1,2)*(x1 - x6)*Power(x6,2)*y5 + Power(x5,3)* (-(Power(x6,2)*y1) + Power(x1,2)*y6) + Power(x5,2)* (Power(x6,3)*y1 - Power(x1,3)*y6)) + Power(x4,2)* (Power(x1,3)*(x1 - x6)*Power(x6,3)*y5 + Power(x5,4)* (-(Power(x6,3)*y1) + Power(x1,3)*y6) + Power(x5,3)* (Power(x6,4)*y1 - Power(x1,4)*y6)) + Power(x4,3)* (Power(x1,2)*Power(x6,2)* (-Power(x1,2) + Power(x6,2))*y5 + Power(x5,4)* (Power(x6,2)*y1 - Power(x1,2)*y6) + Power(x5,2)* (-(Power(x6,4)*y1) + Power(x1,4)*y6))) + Power(x3,2)* (x1*x5*x6*(Power(x5,2)*Power(x6,2)* (-x5 + x6) + Power(x1,3)* (-Power(x5,2) + Power(x6,2)) + Power(x1,2)*(Power(x5,3) - Power(x6,3))) *y4 + Power(x4,4)* (x1*x6*(-Power(x1,2) + Power(x6,2))*y5 + Power(x5,3)*(x6*y1 - x1*y6) + x5*(-(Power(x6,3)*y1) + Power(x1,3)*y6)) + Power(x4,3)* (x1*x6*(Power(x1,3) - Power(x6,3))*y5 + Power(x5,4)*(-(x6*y1) + x1*y6) + x5*(Power(x6,4)*y1 - Power(x1,4)*y6)) + x4*(Power(x1,3)*Power(x6,3)*(-x1 + x6)* y5 + Power(x5,4)* (Power(x6,3)*y1 - Power(x1,3)*y6) + Power(x5,3)* (-(Power(x6,4)*y1) + Power(x1,4)*y6)))) + Power(x2,4)*(x1*(x1 - x4)*x4*(x1 - x5)* (x4 - x5)*x5*(x1 - x6)*(x4 - x6)*(x5 - x6)* x6*(x1 + x4 + x5 + x6)*y3 + Power(x3,5)*(x1*(x1 - x5)*x5*(x1 - x6)* (x5 - x6)*x6*y4 + Power(x4,3)* (x1*(x1 - x6)*x6*y5 + Power(x5,2)*(-(x6*y1) + x1*y6) + x5*(Power(x6,2)*y1 - Power(x1,2)*y6)) + x4*(Power(x1,2)*(x1 - x6)*Power(x6,2)*y5 + Power(x5,3)* (-(Power(x6,2)*y1) + Power(x1,2)*y6) + Power(x5,2)* (Power(x6,3)*y1 - Power(x1,3)*y6)) + Power(x4,2)* (x1*x6*(-Power(x1,2) + Power(x6,2))*y5 + Power(x5,3)*(x6*y1 - x1*y6) + x5*(-(Power(x6,3)*y1) + Power(x1,3)*y6))) + Power(x3,2)* (x1*x5*(Power(x1,2) - Power(x5,2))*x6* (Power(x1,2) - Power(x6,2))* (Power(x5,2) - Power(x6,2))*y4 + Power(x4,5)* (x1*x6*(Power(x1,2) - Power(x6,2))*y5 + Power(x5,3)*(-(x6*y1) + x1*y6) + x5*(Power(x6,3)*y1 - Power(x1,3)*y6)) + x4*(Power(x1,3)*Power(x6,3)* (Power(x1,2) - Power(x6,2))*y5 + Power(x5,5)* (-(Power(x6,3)*y1) + Power(x1,3)*y6) + Power(x5,3)* (Power(x6,5)*y1 - Power(x1,5)*y6)) + Power(x4,3)* (x1*x6*(-Power(x1,4) + Power(x6,4))*y5 + Power(x5,5)*(x6*y1 - x1*y6) + x5*(-(Power(x6,5)*y1) + Power(x1,5)*y6))) + Power(x3,3)* (x1*x5*x6*(-(Power(x5,4)*x6) + x5*Power(x6,4) + Power(x1,4)*(-x5 + x6) + x1*(Power(x5,4) - Power(x6,4)))*y4 + Power(x4,5)* (x1*x6*(-x1 + x6)*y5 + Power(x5,2)*(x6*y1 - x1*y6) + x5*(-(Power(x6,2)*y1) + Power(x1,2)*y6)) + Power(x4,2)* (x1*x6*(Power(x1,4) - Power(x6,4))*y5 + Power(x5,5)*(-(x6*y1) + x1*y6) + x5*(Power(x6,5)*y1 - Power(x1,5)*y6)) + x4*(Power(x1,2)*Power(x6,2)* (-Power(x1,3) + Power(x6,3))*y5 + Power(x5,5)* (Power(x6,2)*y1 - Power(x1,2)*y6) + Power(x5,2)* (-(Power(x6,5)*y1) + Power(x1,5)*y6))) + x3*(Power(x1,2)*Power(x5,2)*Power(x6,2)* (-(Power(x5,3)*x6) + x5*Power(x6,3) + Power(x1,3)*(-x5 + x6) + x1*(Power(x5,3) - Power(x6,3)))*y4 + Power(x4,5)* (Power(x1,2)*Power(x6,2)*(-x1 + x6)*y5 + Power(x5,3)* (Power(x6,2)*y1 - Power(x1,2)*y6) + Power(x5,2)* (-(Power(x6,3)*y1) + Power(x1,3)*y6)) + Power(x4,3)* (Power(x1,2)*Power(x6,2)* (Power(x1,3) - Power(x6,3))*y5 + Power(x5,5)* (-(Power(x6,2)*y1) + Power(x1,2)*y6) + Power(x5,2)* (Power(x6,5)*y1 - Power(x1,5)*y6)) + Power(x4,2)* (Power(x1,3)*Power(x6,3)* (-Power(x1,2) + Power(x6,2))*y5 + Power(x5,5)* (Power(x6,3)*y1 - Power(x1,3)*y6) + Power(x5,3)* (-(Power(x6,5)*y1) + Power(x1,5)*y6)))) + Power(x2,3)*(-(x1*(x1 - x4)*x4*(x1 - x5)* (x4 - x5)*x5*(x1 - x6)*(x4 - x6)*(x5 - x6)* x6*(x5*x6 + x4*(x5 + x6) + x1*(x4 + x5 + x6))*y3) + Power(x3,5)*(x1*x5*x6* (-(Power(x5,3)*x6) + x5*Power(x6,3) + Power(x1,3)*(-x5 + x6) + x1*(Power(x5,3) - Power(x6,3)))*y4 + Power(x4,4)* (x1*x6*(-x1 + x6)*y5 + Power(x5,2)*(x6*y1 - x1*y6) + x5*(-(Power(x6,2)*y1) + Power(x1,2)*y6)) + Power(x4,2)* (x1*x6*(Power(x1,3) - Power(x6,3))*y5 + Power(x5,4)*(-(x6*y1) + x1*y6) + x5*(Power(x6,4)*y1 - Power(x1,4)*y6)) + x4*(Power(x1,2)*Power(x6,2)* (-Power(x1,2) + Power(x6,2))*y5 + Power(x5,4)* (Power(x6,2)*y1 - Power(x1,2)*y6) + Power(x5,2)* (-(Power(x6,4)*y1) + Power(x1,4)*y6))) + Power(x3,4)* (x1*x5*x6*(Power(x1,4)*(x5 - x6) + x5*x6*(Power(x5,3) - Power(x6,3)) + x1*(-Power(x5,4) + Power(x6,4)))*y4 + Power(x4,5)* (x1*(x1 - x6)*x6*y5 + Power(x5,2)*(-(x6*y1) + x1*y6) + x5*(Power(x6,2)*y1 - Power(x1,2)*y6)) + x4*(Power(x1,2)*Power(x6,2)* (Power(x1,3) - Power(x6,3))*y5 + Power(x5,5)* (-(Power(x6,2)*y1) + Power(x1,2)*y6) + Power(x5,2)* (Power(x6,5)*y1 - Power(x1,5)*y6)) + Power(x4,2)* (x1*x6*(-Power(x1,4) + Power(x6,4))*y5 + Power(x5,5)*(x6*y1 - x1*y6) + x5*(-(Power(x6,5)*y1) + Power(x1,5)*y6))) + x3*(Power(x1,2)*Power(x5,2)* Power(x6,2)* (Power(x5,2)*(x5 - x6)*Power(x6,2) + Power(x1,3)* (Power(x5,2) - Power(x6,2)) + Power(x1,2)*(-Power(x5,3) + Power(x6,3)))*y4 + Power(x4,5)* (Power(x1,2)*Power(x6,2)* (Power(x1,2) - Power(x6,2))*y5 + Power(x5,4)* (-(Power(x6,2)*y1) + Power(x1,2)*y6) + Power(x5,2)* (Power(x6,4)*y1 - Power(x1,4)*y6)) + Power(x4,2)* (Power(x1,4)*(x1 - x6)*Power(x6,4)*y5 + Power(x5,5)* (-(Power(x6,4)*y1) + Power(x1,4)*y6) + Power(x5,4)* (Power(x6,5)*y1 - Power(x1,5)*y6)) + Power(x4,4)* (Power(x1,2)*Power(x6,2)* (-Power(x1,3) + Power(x6,3))*y5 + Power(x5,5)* (Power(x6,2)*y1 - Power(x1,2)*y6) + Power(x5,2)* (-(Power(x6,5)*y1) + Power(x1,5)*y6))) + Power(x3,2)* (x1*x5*x6*(Power(x5,3)*Power(x6,3)* (-x5 + x6) + Power(x1,4)* (-Power(x5,3) + Power(x6,3)) + Power(x1,3)*(Power(x5,4) - Power(x6,4))) *y4 + Power(x4,5)* (x1*x6*(-Power(x1,3) + Power(x6,3))*y5 + Power(x5,4)*(x6*y1 - x1*y6) + x5*(-(Power(x6,4)*y1) + Power(x1,4)*y6)) + Power(x4,4)* (x1*x6*(Power(x1,4) - Power(x6,4))*y5 + Power(x5,5)*(-(x6*y1) + x1*y6) + x5*(Power(x6,5)*y1 - Power(x1,5)*y6)) + x4*(Power(x1,4)*Power(x6,4)*(-x1 + x6)* y5 + Power(x5,5)* (Power(x6,4)*y1 - Power(x1,4)*y6) + Power(x5,4)* (-(Power(x6,5)*y1) + Power(x1,5)*y6)))) + x2*(-(Power(x1,2)*(x1 - x4)*Power(x4,2)* (x1 - x5)*(x4 - x5)*Power(x5,2)*(x1 - x6)* (x4 - x6)*(x5 - x6)*Power(x6,2)*y3) + Power(x3,5)*(-(Power(x1,2)*(x1 - x5)* Power(x5,2)*(x1 - x6)*(x5 - x6)* Power(x6,2)*y4) + Power(x4,4)* (Power(x1,2)*Power(x6,2)*(-x1 + x6)*y5 + Power(x5,3)* (Power(x6,2)*y1 - Power(x1,2)*y6) + Power(x5,2)* (-(Power(x6,3)*y1) + Power(x1,3)*y6)) + Power(x4,3)* (Power(x1,2)*Power(x6,2)* (Power(x1,2) - Power(x6,2))*y5 + Power(x5,4)* (-(Power(x6,2)*y1) + Power(x1,2)*y6) + Power(x5,2)* (Power(x6,4)*y1 - Power(x1,4)*y6)) + Power(x4,2)* (Power(x1,3)*Power(x6,3)*(-x1 + x6)*y5 + Power(x5,4)* (Power(x6,3)*y1 - Power(x1,3)*y6) + Power(x5,3)* (-(Power(x6,4)*y1) + Power(x1,4)*y6))) + Power(x3,4)* (Power(x1,2)*Power(x5,2)*Power(x6,2)* (Power(x1,3)*(x5 - x6) + x5*x6*(Power(x5,2) - Power(x6,2)) + x1*(-Power(x5,3) + Power(x6,3)))*y4 + Power(x4,5)* (Power(x1,2)*(x1 - x6)*Power(x6,2)*y5 + Power(x5,3)* (-(Power(x6,2)*y1) + Power(x1,2)*y6) + Power(x5,2)* (Power(x6,3)*y1 - Power(x1,3)*y6)) + Power(x4,2)* (Power(x1,3)*Power(x6,3)* (Power(x1,2) - Power(x6,2))*y5 + Power(x5,5)* (-(Power(x6,3)*y1) + Power(x1,3)*y6) + Power(x5,3)* (Power(x6,5)*y1 - Power(x1,5)*y6)) + Power(x4,3)* (Power(x1,2)*Power(x6,2)* (-Power(x1,3) + Power(x6,3))*y5 + Power(x5,5)* (Power(x6,2)*y1 - Power(x1,2)*y6) + Power(x5,2)* (-(Power(x6,5)*y1) + Power(x1,5)*y6))) + Power(x3,2)* (Power(x1,3)*(x1 - x5)*Power(x5,3)*(x1 - x6)* (x5 - x6)*Power(x6,3)*y4 + Power(x4,5)* (Power(x1,3)*(x1 - x6)*Power(x6,3)*y5 + Power(x5,4)* (-(Power(x6,3)*y1) + Power(x1,3)*y6) + Power(x5,3)* (Power(x6,4)*y1 - Power(x1,4)*y6)) + Power(x4,3)* (Power(x1,4)*(x1 - x6)*Power(x6,4)*y5 + Power(x5,5)* (-(Power(x6,4)*y1) + Power(x1,4)*y6) + Power(x5,4)* (Power(x6,5)*y1 - Power(x1,5)*y6)) + Power(x4,4)* (Power(x1,3)*Power(x6,3)* (-Power(x1,2) + Power(x6,2))*y5 + Power(x5,5)* (Power(x6,3)*y1 - Power(x1,3)*y6) + Power(x5,3)* (-(Power(x6,5)*y1) + Power(x1,5)*y6))) + Power(x3,3)* (Power(x1,2)*Power(x5,2)*Power(x6,2)* (Power(x5,2)*Power(x6,2)*(-x5 + x6) + Power(x1,3)* (-Power(x5,2) + Power(x6,2)) + Power(x1,2)*(Power(x5,3) - Power(x6,3))) *y4 + Power(x4,5)* (Power(x1,2)*Power(x6,2)* (-Power(x1,2) + Power(x6,2))*y5 + Power(x5,4)* (Power(x6,2)*y1 - Power(x1,2)*y6) + Power(x5,2)* (-(Power(x6,4)*y1) + Power(x1,4)*y6)) + Power(x4,4)* (Power(x1,2)*Power(x6,2)* (Power(x1,3) - Power(x6,3))*y5 + Power(x5,5)* (-(Power(x6,2)*y1) + Power(x1,2)*y6) + Power(x5,2)* (Power(x6,5)*y1 - Power(x1,5)*y6)) + Power(x4,2)* (Power(x1,4)*Power(x6,4)*(-x1 + x6)*y5 + Power(x5,5)* (Power(x6,4)*y1 - Power(x1,4)*y6) + Power(x5,4)* (-(Power(x6,5)*y1) + Power(x1,5)*y6)))) + Power(x2,2)*(x1*(x1 - x4)*x4*(x1 - x5)* (x4 - x5)*x5*(x1 - x6)*(x4 - x6)*(x5 - x6)* x6*(x4*x5*x6 + x1*(x5*x6 + x4*(x5 + x6)))*y3 + Power(x3,5)* (x1*x5*x6*(Power(x5,2)*(x5 - x6)* Power(x6,2) + Power(x1,3)* (Power(x5,2) - Power(x6,2)) + Power(x1,2)*(-Power(x5,3) + Power(x6,3)))*y4 + Power(x4,4)* (x1*x6*(Power(x1,2) - Power(x6,2))*y5 + Power(x5,3)*(-(x6*y1) + x1*y6) + x5*(Power(x6,3)*y1 - Power(x1,3)*y6)) + x4*(Power(x1,3)*(x1 - x6)*Power(x6,3)*y5 + Power(x5,4)* (-(Power(x6,3)*y1) + Power(x1,3)*y6) + Power(x5,3)* (Power(x6,4)*y1 - Power(x1,4)*y6)) + Power(x4,3)* (x1*x6*(-Power(x1,3) + Power(x6,3))*y5 + Power(x5,4)*(x6*y1 - x1*y6) + x5*(-(Power(x6,4)*y1) + Power(x1,4)*y6))) + Power(x3,3)* (x1*x5*x6*(Power(x5,3)*(x5 - x6)* Power(x6,3) + Power(x1,4)* (Power(x5,3) - Power(x6,3)) + Power(x1,3)*(-Power(x5,4) + Power(x6,4)))*y4 + Power(x4,5)* (x1*x6*(Power(x1,3) - Power(x6,3))*y5 + Power(x5,4)*(-(x6*y1) + x1*y6) + x5*(Power(x6,4)*y1 - Power(x1,4)*y6)) + x4*(Power(x1,4)*(x1 - x6)*Power(x6,4)*y5 + Power(x5,5)* (-(Power(x6,4)*y1) + Power(x1,4)*y6) + Power(x5,4)* (Power(x6,5)*y1 - Power(x1,5)*y6)) + Power(x4,4)* (x1*x6*(-Power(x1,4) + Power(x6,4))*y5 + Power(x5,5)*(x6*y1 - x1*y6) + x5*(-(Power(x6,5)*y1) + Power(x1,5)*y6))) + Power(x3,4)* (-(x1*x5*(Power(x1,2) - Power(x5,2))*x6* (Power(x1,2) - Power(x6,2))* (Power(x5,2) - Power(x6,2))*y4) + Power(x4,5)* (x1*x6*(-Power(x1,2) + Power(x6,2))*y5 + Power(x5,3)*(x6*y1 - x1*y6) + x5*(-(Power(x6,3)*y1) + Power(x1,3)*y6)) + Power(x4,3)* (x1*x6*(Power(x1,4) - Power(x6,4))*y5 + Power(x5,5)*(-(x6*y1) + x1*y6) + x5*(Power(x6,5)*y1 - Power(x1,5)*y6)) + x4*(Power(x1,3)*Power(x6,3)* (-Power(x1,2) + Power(x6,2))*y5 + Power(x5,5)* (Power(x6,3)*y1 - Power(x1,3)*y6) + Power(x5,3)* (-(Power(x6,5)*y1) + Power(x1,5)*y6))) + x3*(-(Power(x1,3)*(x1 - x5)*Power(x5,3)* (x1 - x6)*(x5 - x6)*Power(x6,3)*y4) + Power(x4,5)* (Power(x1,3)*Power(x6,3)*(-x1 + x6)*y5 + Power(x5,4)* (Power(x6,3)*y1 - Power(x1,3)*y6) + Power(x5,3)* (-(Power(x6,4)*y1) + Power(x1,4)*y6)) + Power(x4,4)* (Power(x1,3)*Power(x6,3)* (Power(x1,2) - Power(x6,2))*y5 + Power(x5,5)* (-(Power(x6,3)*y1) + Power(x1,3)*y6) + Power(x5,3)* (Power(x6,5)*y1 - Power(x1,5)*y6)) + Power(x4,3)* (Power(x1,4)*Power(x6,4)*(-x1 + x6)*y5 + Power(x5,5)* (Power(x6,4)*y1 - Power(x1,4)*y6) + Power(x5,4)* (-(Power(x6,5)*y1) + Power(x1,5)*y6)))))/((x1 - x2)*(x1 - x3)*(-x2 + x3)*(x1 - x4)* (-x2 + x4)*(-x3 + x4)*(x1 - x5)*(x2 - x5)* (x3 - x5)*(x4 - x5)*(x1 - x6)*(x2 - x6)* (x3 - x6)*(x4 - x6)*(x5 - x6))

def reverse_array(arr):
    """
    Reverse an array along all axes, so arr[i,j] -> arr[-(i+1),-(j+1)].
    """
    reverse_slice = [slice(None, None, -1) for ii in arr.shape]
    return arr[reverse_slice]

def intersect_masks(m1, m2):
    """
    Versions of m1 and m2 that are masked where either m1 or m2 were masked.

    If neither m1 or m2 is masked, just returns m1 and m2. Otherwise returns
    m1 and m2 wrapped as masked_arrays with identical masks.
    """
    ma = numpy.ma
    import dadi
    if ma.isMaskedArray(m1) or ma.isMaskedArray(m2):
        joint_mask = ma.mask_or(ma.getmask(m1), ma.getmask(m2))

        m1 = dadi.Spectrum(m1, mask=joint_mask.copy())
        m2 = dadi.Spectrum(m2, mask=joint_mask.copy())
    return m1,m2

def trapz(yy, xx=None, dx=None, axis=-1):
    """
    Integrate yy(xx) along given axis using the composite trapezoidal rule.
    
    xx must be one-dimensional and len(xx) must equal yy.shape[axis].

    This is modified from the SciPy version to work with n-D yy and 1-D xx.
    """
    if (xx is None and dx is None)\
       or (xx is not None and dx is not None):
        raise ValueError('One and only one of xx or dx must be specified.')
    elif (xx is not None) and (dx is None):
        dx = numpy.diff(xx)
    yy = numpy.asanyarray(yy)
    nd = yy.ndim

    if yy.shape[axis] != (len(dx)+1):
        raise ValueError('Length of xx must be equal to length of yy along '
                         'specified axis. Here len(xx) = %i and '
                         'yy.shape[axis] = %i.' % (len(dx)+1, yy.shape[axis]))

    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1,None)
    slice2[axis] = slice(None,-1)
    sliceX = [numpy.newaxis]*nd
    sliceX[axis] = slice(None)

    return numpy.sum(dx[sliceX] * (yy[slice1]+yy[slice2])/2.0, axis=axis)

def make_extrap_func(func, extrap_x_l=None, extrap_log=False, fail_mag=10):
    """
    Generate a version of func that extrapolates to infinitely many gridpoints.

    func: A function that returns a single scalar or array and whose last
        non-keyword argument is 'pts': the number of default_grid points to use
        in calculation.  
    extrap_x_l: An explict list of x values to use for extrapolation. If not 
        provided, the extrapolation routine will look for '.extrap_x'
        attributes on the results of func. The method Spectrum.from_phi will
        add an extrap_x attribute to resulting Spectra, equal to the x-value
        of the first non-zero grid point. An explicit list is useful if you
        want to override this behavior for testing.
    fail_mag:  Simon Gravel noted that there can be numerical instabilities in
        extrapolation when working with large spectra that have very small
        entires (of order 1e-24). To avoid these instabilities, we ignore the 
        extrapolation values (and use the input result with the smallest x) 
        if the extrapolation is more than fail_mag orders of magnitude away
        from the smallest x input result.

    Returns a new function whose last argument is a list of numbers of grid
    points and that returns a result extrapolated to infinitely many grid
    points.
    """
    x_l_from_results = (extrap_x_l is None)

    def extrap_func(*args, **kwargs):
        # Separate pts (or pts_l) from arguments
        if 'pts' not in kwargs:
            other_args, pts_l = args[:-1], args[-1]
        else:
            other_args = args
            pts_l = kwargs['pts']
            del kwargs['pts']

        if 'no_extrap' in kwargs:
            no_extrap = True
            del kwargs['no_extrap']
        else:
            no_extrap = False

        if numpy.isscalar(pts_l):
            pts_l = [pts_l]

        # Create a sub-function that fixes all other arguments and only
        # takes in pts.
        partial_func = functools.partial(func, *other_args, **kwargs)

        ##
        ## The commented-out code implements distribution of fs calculations
        ## among multiple processors. Unfortunately, it doesn't work with
        ## iPython, because pickling functions is very fragile in the
        ## interactive interpreter. If we had some sort of data structure for
        ## defining models, then this would be much easier to implement. Note
        ## that there might still be issues on Windows systems, because of its
        ## poor-man's version of fork().
        ##
        #import cPickle, multiprocessing
        #try:
        #    # Test whether sub-function is picklable. If not, pool.map will
        #    # hang.
        #    cPickle.dumps(partial_func)
        #    pool = multiprocessing.Pool()
        #    result_l = pool.map(partial_func, pts_l)
        #    pool.close()
        #except (cPickle.PicklingError, TypeError):
        #    print('Function passed to extrap func must be picklable for '
        #          'multi-processor executation.')
        #    import sys
        #    sys.exit()

        result_l = map(partial_func, pts_l)
        if no_extrap:
            return result_l

        if x_l_from_results:
            try:
                x_l = [r.extrap_x for r in result_l]
            except AttributeError:
                raise ValueError("Extrapolation function error: No explicit extrapolation x_l provided, and results do not have 'extrap_x' attributes. If this is an FS extrapolation, check your from_phi method.")
        else:
            x_l = extrap_x_l

        if extrap_log:
            result_l = [numpy.log(r) for r in result_l]

        # Extrapolate
        if len(pts_l) == 1:
            ex_result = result_l[0]
        elif len(pts_l) == 2:
            ex_result = linear_extrap(result_l, x_l)
        elif len(pts_l) == 3:
            ex_result = quadratic_extrap(result_l, x_l)
        elif len(pts_l) == 4:
            ex_result = cubic_extrap(result_l, x_l)
        elif len(pts_l) == 5:
            ex_result = quartic_extrap(result_l, x_l)
        elif len(pts_l) == 6:
            ex_result = quintic_extrap(result_l, x_l)
        else:
            raise ValueError('Number of calculations to use for extrapolation '
                             'must be between 1 and 6')

        if extrap_log:
            ex_result = numpy.exp(ex_result)

        # Simon Gravel noted that there can be numerical instabilities in
        # extrapolation when working with large spectra that have very small
        # entires (of order 1e-24).
        # To avoid these instabilities, we ignore the extrapolation values
        # if it is too different from the input values.
        if len(pts_l) > 1:
            # Assume the best input value comes from the smallest grid.
            best_result = result_l[numpy.argmin(x_l)]
            if extrap_log:
                best_result = numpy.exp(best_result)

            # The extrapolation is deemed to have failed if it results in a
            # value more than fail_mag orders of magnitude away from the 
            # best input value.
            extrap_failed = abs(numpy.log10(ex_result/best_result)) > fail_mag
            if numpy.any(extrap_failed):
                logger.warn('Extrapolation may have failed. Check resulting '
                            'frequency spectrum for unexpected results.')

            # For entries that fail, use the "best" input result.
            ex_result[extrap_failed] = best_result[extrap_failed]

        return ex_result

    extrap_func.func_name = func.func_name
    extrap_func.func_doc = func.func_doc

    return extrap_func

def make_extrap_log_func(func, extrap_x_l=None):
    """
    Generate a version of func that extrapolates to infinitely many gridpoints.

    Note that extrapolation here is done on the *log* of the function result,
    so this will fail if any returned values are < 0. It does seem to be better
    behaved for SFS calculation.

    func: A function whose last argument is the number of Numerics.default_grid 
          points to use in calculation and that returns a single scalar or 
          array.
    extrap_x_l: An explict list of x values to use for extrapolation. If not 
         provided, the extrapolation routine will look for '.extrap_x'
         attributes on the results of func. The method Spectrum.from_phi will
         add an extrap_x attribute to resulting Spectra, equal to the x-value
         of the first non-zero grid point. An explicit list is useful if you
         want to override this behavior for testing.

    Returns a new function whose last argument is a list of numbers of grid
    points and that returns a result extrapolated to infinitely many grid
    points.
    """
    return make_extrap_func(func, extrap_x_l=extrap_x_l, extrap_log=True)

_projection_cache = {}
def _lncomb(N,k):
    """
    Log of N choose k.
    """
    return gammaln(N+1) - gammaln(k+1) - gammaln(N-k+1)

def _cached_projection(proj_to, proj_from, hits):
    """
    Coefficients for projection from a different fs size.

    proj_to: Numper of samples to project down to.
    proj_from: Numper of samples to project from.
    hits: Number of derived alleles projecting from.
    """
    key = (proj_to, proj_from, hits)
    try:
        return _projection_cache[key]
    except KeyError:
        pass

    if numpy.isscalar(proj_to) and numpy.isscalar(proj_from)\
       and proj_from < proj_to:
        # Short-circuit calculation.
        contrib = numpy.zeros(proj_to+1)
    else:
        # We set numpy's error reporting so that it will ignore underflows, 
        # because those just imply that contrib is 0.
        previous_err_state = numpy.seterr(under='ignore', divide='raise',
                                          over='raise', invalid='raise')
        proj_hits = numpy.arange(proj_to+1)
        # For large sample sizes, we need to do the calculation in logs, and it
        # is accurate enough for small sizes as well.
        lncontrib = _lncomb(proj_to,proj_hits)
        lncontrib += _lncomb(proj_from-proj_to,hits-proj_hits)
        lncontrib -= _lncomb(proj_from, hits)
        contrib = numpy.exp(lncontrib)
        numpy.seterr(**previous_err_state)
    _projection_cache[key] = contrib
    return contrib

def array_from_file(fid, return_comments=False):
    """
    Read array from file.

    fid: string with file name to read from or an open file object.
    return_comments: If True, the return value is (fs, comments), where
                     comments is a list of strings containing the comments
                     from the file (without #'s).

    The file format is:
        # Any number of comment lines beginning with a '#'
        A single line containing N integers giving the dimensions of the fs
          array. So this line would be '5 5 3' for an SFS that was 5x5x3.
          (That would be 4x4x2 *samples*.)
        A single line giving the array elements. The order of elements is 
          e.g.: fs[0,0,0] fs[0,0,1] fs[0,0,2] ... fs[0,1,0] fs[0,1,1] ...
    """
    newfile = False
    # Try to read from fid. If we can't, assume it's something that we can
    # use to open a file.
    if not hasattr(fid, 'read'):
        newfile = True
        fid = file(fid, 'r')

    line = fid.readline()
    # Strip out the comments
    comments = []
    while line.startswith('#'):
        comments.append(line[1:].strip())
        line = fid.readline()

    # Read the shape of the data
    shape = tuple([int(d) for d in line.split()])

    data = numpy.fromfile(fid, count=numpy.product(shape), sep=' ')
    # fromfile returns a 1-d array. Reshape it to the proper form.
    data = data.reshape(*shape)

    # If we opened a new file, clean it up.
    if newfile:
        fid.close()

    if not return_comments:
        return data
    else:
        return data,comments

def array_to_file(data, fid, precision=16, comment_lines = []):
    """
    Write array to file.

    data: array to write
    fid: string with file name to write to or an open file object.
    precision: precision with which to write out entries of the SFS. (They 
               are formated via %.<p>g, where <p> is the precision.)
    comment lines: list of strings to be used as comment lines in the header
                   of the output file.

    The file format is:
        # Any number of comment lines beginning with a '#'
        A single line containing N integers giving the dimensions of the fs
          array. So this line would be '5 5 3' for an SFS that was 5x5x3.
          (That would be 4x4x2 *samples*.)
        A single line giving the array elements. The order of elements is 
          e.g.: fs[0,0,0] fs[0,0,1] fs[0,0,2] ... fs[0,1,0] fs[0,1,1] ...
    """
    # Open the file object.
    newfile = False
    if not hasattr(fid, 'write'):
        newfile = True
        fid = file(fid, 'w')

    # Write comments
    for line in comment_lines:
        fid.write('# ')
        fid.write(line.strip())
        fid.write(os.linesep)

    # Write out the shape of the fs
    for elem in data.shape:
        fid.write('%i ' % elem)
    fid.write(os.linesep)

    if hasattr(data, 'filled'):
        # Masked entries in the fs will go in as 'nan'
        data = data.filled()
    # Write to file
    data.tofile(fid, ' ', '%%.%ig' % precision)
    fid.write(os.linesep)

    # Close file
    if newfile:
        fid.close()
