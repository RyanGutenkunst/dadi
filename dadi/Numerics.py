"""
Numerically useful functions, including extrapolation and default grid.
"""

import numpy
from scipy import comb
from scipy.special import gammaln

def default_grid(num_pts):
    """
    A nonuniform grid of points on [0,1].

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

def reverse_array(arr):
    """
    Reverse an array along all axes, so arr[i,j] -> arr[-(i+1),-(j+1)].
    """
    reverse_slice = [slice(None, None, -1) for ii in arr.shape]
    return arr[reverse_slice]

def intersect_masks(m1, m2):
    """
    Versions of m1 and m2 that are masked where either m1 or m2 were masked.

    If neither m1 or m2 is masked, just returns m1 or m2. Otherwise returns
    m1 and m2 wrapped as masked_arrays with identical masks.
    """
    ma = numpy.ma
    if ma.isMaskedArray(m1) or ma.isMaskedArray(m2):
        joint_mask = ma.mask_or(ma.getmask(m1), ma.getmask(m2))

        m1 = ma.masked_array(m1, mask=joint_mask.copy())
        m2 = ma.masked_array(m2, mask=joint_mask.copy())
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
    nd = yy.ndim

    yy = numpy.asarray(yy)
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

def make_extrap_func(func):
    """
    Generate a version of func that extrapolates to infinitely many gridpoints.

    func: A function whose last argument is the number of default_grid
          points to use in calculation and that returns a single scalar or 
          array.

    Returns a new function whose last argument is a list of numbers of grid
    points and that returns a result extrapolated to infinitely many grid
    points.
    """
    def extrap_func(*args):
        other_args, pts_l = args[:-1], args[-1]

        x_l, result_l = [],[]
        for pts in pts_l:
            # We extrapolate based on the first grid spacing. This seems to
            # give better results than, for example, the average spacing.
            x = default_grid(pts)[1]
            x_l.append(x)
            # Some python vodoo here to call the original function with the
            # proper arguments.
            result = func(*(other_args + (pts,)))
            result_l.append(result)

        # Extrapolate
        if len(pts_l) == 1:
            ex_result = result_l[0]
        elif len(pts_l) == 2:
            ex_result = linear_extrap(result_l, x_l)
        elif len(pts_l) == 3:
            ex_result = quadratic_extrap(result_l, x_l)
        elif len(pts_l) == 4:
            ex_result = cubic_extrap(result_l, x_l)
        else:
            raise ValueError('Number of calculations to use for extrapolation '
                             'must be between 1 and 4')
        return ex_result

    return extrap_func

def make_extrap_log_func(func):
    """
    Generate a version of func that extrapolates to infinitely many gridpoints.

    Note that extrapolation here is done on the *log* of the function result,
    so this will fail if any returned values are < 0. It does seem to be better
    behaved for SFS calculation.

    func: A function whose last argument is the number of Numerics.default_grid 
          points to use in calculation and that returns a single scalar or 
          array.

    Returns a new function whose last argument is a list of numbers of grid
    points and that returns a result extrapolated to infinitely many grid
    points.
    """
    def logfunc(*args):
        return numpy.log(func(*args))
    exlog_func = make_extrap_func(logfunc)
    def ex_func(*args):
        return numpy.exp(exlog_func(*args))
    return ex_func

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

    proj_hits = numpy.arange(proj_to+1)
    contrib = comb(proj_to,proj_hits)*comb(proj_from-proj_to,hits-proj_hits)
    contrib /= comb(proj_from, hits)

    if numpy.any(numpy.isnan(contrib)):
        lncontrib = _lncomb(proj_to,proj_hits)
        lncontrib += _lncomb(proj_from-proj_to,hits-proj_hits)
        lncontrib -= _lncomb(proj_from, hits)
        contrib = numpy.exp(lncontrib)

    _projection_cache[key] = contrib
    return contrib
