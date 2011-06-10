"""
Functions for integrating population frequency spectra.
"""

import logging
logger = logging.getLogger('Integration')

#
# Note that these functions have all be written for xx=yy=zz, so the grids are
# identical in each direction. That's not essential here, though. The reason we
# do it is because later we'll want to extrapolate, and it's not obvious how to
# do so if the grids differ.
#

#: Controls use of Chang and Cooper's delj trick, which seems to lower accuracy.
use_delj_trick = False

import numpy
from numpy import newaxis as nuax

import Misc, Numerics, tridiag
import integration_c as int_c

#: Controls timestep for integrations. This is a reasonable default for
#: gridsizes of ~60. See set_timescale_factor for better control.
timescale_factor = 1e-3

#: Whether to use old timestep method, which is old_timescale_factor * dx[0].
use_old_timestep = False
#: Factor for told timestep method.
old_timescale_factor = 0.1

def set_timescale_factor(pts, factor=10):
    """
    Controls the fineness of timesteps during integration.

    The timestep will be proportional to Numerics.default_grid(pts)[1]/factor.
    Typically, pts should be set to the *largest* number of grid points used in
    extrapolation. 
    
    An adjustment factor of 10 typically results in acceptable accuracy. It may
    be desirable to increase this factor, particularly when population sizes
    are changing continously and rapidly.
    """
    # Implementation note: This cannot be easily be set automatically, because
    # the integration doesn't know whether its results will be used in an
    # extrapolation.
    global timescale_factor
    logger.warn('set_timescale_factor has been deprecated, as it may be too '
                'conservative (and thus slow) in choosing timesteps. If you '
                'wish to take smaller timesteps for accuracy (particularly for '
                'a very quickly growing population), manually set '
                'dadi.Integration.timescale_factor to a smaller value. '
                '(Current value is %g.)' % timescale_factor)
    timescale_factor = Numerics.default_grid(pts)[1]/factor

def _inject_mutations_1D(phi, dt, xx, theta0):
    """
    Inject novel mutations for a timestep.
    """
    phi[1] += dt/xx[1] * theta0/2 * 2/(xx[2] - xx[0])
    return phi
def _inject_mutations_2D(phi, dt, xx, yy, theta0, frozen1, frozen2):
    """
    Inject novel mutations for a timestep.
    """
    # Population 1
    if not frozen1:
        phi[1,0] += dt/xx[1] * theta0/2 * 4/((xx[2] - xx[0]) * yy[1])
    # Population 2
    if not frozen2:
        phi[0,1] += dt/yy[1] * theta0/2 * 4/((yy[2] - yy[0]) * xx[1])
    return phi
def _inject_mutations_3D(phi, dt, xx, yy, zz, theta0, frozen1, frozen2, frozen3):
    """
    Inject novel mutations for a timestep.
    """
    # Population 1
    # Normalization based on the multi-dimensional trapezoid rule is 
    # implemented                      ************** here ***************
    if not frozen1:
        phi[1,0,0] += dt/xx[1] * theta0/2 * 8/((xx[2] - xx[0]) * yy[1] * zz[1])
    # Population 2
    if not frozen2:
        phi[0,1,0] += dt/yy[1] * theta0/2 * 8/((yy[2] - yy[0]) * xx[1] * zz[1])
    # Population 3
    if not frozen3:
        phi[0,0,1] += dt/zz[1] * theta0/2 * 8/((zz[2] - zz[0]) * xx[1] * yy[1])
    return phi

def _compute_dt(dx, nu, ms, gamma, h):
    """
    Compute the appropriate timestep given the current demographic params.

    This is based on the maximum V or M expected in this direction. The
    timestep is scaled such that if the params are rescaled correctly by a
    constant, the exact same integration happens. (This is equivalent to
    multiplying the eqn through by some other 2N...)
    """
    if use_old_timestep:
        return old_timescale_factor * dx[0]

    # These are the maxima for V_func and M_func over the domain
    # For h != 0.5, the maximum of M_func is not easy analytically. It is close
    # to the 0.5 or 0.25 value, though, so we use those as an approximation.

    # It might seem natural to scale dt based on dx[0]. However, testing has
    # shown that extrapolation is much more reliable when the same timesteps
    # are used in evaluations at different grid sizes.
    maxVM = max(0.25/nu, sum(ms),\
                abs(gamma) * 2*max(numpy.abs(h + (1-2*h)*0.5) * 0.5*(1-0.5),
                                   numpy.abs(h + (1-2*h)*0.25) * 0.25*(1-0.25)))
    if maxVM > 0:
        dt = timescale_factor / maxVM
    else:
        dt = numpy.inf
    if dt == 0:
        raise ValueError('Timestep is zero. Values passed in are nu=%f, ms=%s,'
                         'gamma=%f, h=%f.' % (nu, str(ms), gamma, h))
    return dt

def one_pop(phi, xx, T, nu=1, gamma=0, h=0.5, theta0=1.0, initial_t=0, 
            frozen=False):
    """
    Integrate a 1-dimensional phi foward.

    phi: Initial 1-dimensional phi
    xx: Grid upon (0,1) overwhich phi is defined.

    nu, gamma, and theta0 may be functions of time.
    nu: Population size
    gamma: Selection coefficient on *all* segregating alleles
    h: Dominance coefficient. h = 0.5 corresponds to genic selection.
    theta0: Propotional to ancestral size. Typically constant.

    T: Time at which to halt integration
    initial_t: Time at which to start integration. (Note that this only matters
               if one of the demographic parameters is a function of time.)

    frozen: If True, population is 'frozen' so that it does not change.
            In the one_pop case, this is equivalent to not running the
            integration at all.
    """
    phi = phi.copy()

    # For a one population integration, freezing means just not integrating.
    if frozen:
        return phi

    if T - initial_t == 0:
        return phi
    elif T - initial_t < 0:
        raise ValueError('Final integration time T (%f) is less than '
                         'intial_time (%f). Integration cannot be run '
                         'backwards.' % (T, initial_t))

    vars_to_check = (nu, gamma, h, theta0)
    if numpy.all([numpy.isscalar(var) for var in vars_to_check]):
        return _one_pop_const_params(phi, xx, T, nu, gamma, h, theta0, 
                                     initial_t)

    nu_f = Misc.ensure_1arg_func(nu)
    gamma_f = Misc.ensure_1arg_func(gamma)
    h_f = Misc.ensure_1arg_func(h)
    theta0_f = Misc.ensure_1arg_func(theta0)

    current_t = initial_t
    nu, gamma, h = nu_f(current_t), gamma_f(current_t), h_f(current_t)
    dx = numpy.diff(xx)
    while current_t < T:
        dt = _compute_dt(dx,nu,[0],gamma,h)
        this_dt = min(dt, T - current_t)

        # Because this is an implicit method, I need the *next* time's params.
        # So there's a little inconsistency here, in that I'm estimating dt
        # using the last timepoints nu,gamma,h.
        next_t = current_t + this_dt
        nu, gamma, h = nu_f(next_t), gamma_f(next_t), h_f(next_t)
        theta0 = theta0_f(next_t)

        if numpy.any(numpy.less([T,nu,theta0], 0)):
            raise ValueError('A time, population size, migration rate, or '
                             'theta0 is < 0. Has the model been mis-specified?')
        if numpy.any(numpy.equal([nu], 0)):
            raise ValueError('A population size is 0. Has the model been '
                             'mis-specified?')

        _inject_mutations_1D(phi, this_dt, xx, theta0)
        # Do each step in C, since it will be faster to compute the a,b,c
        # matrices there.
        phi = int_c.implicit_1Dx(phi, xx, nu, gamma, h, this_dt, 
                                 use_delj_trick=use_delj_trick)
        current_t = next_t
    return phi

def two_pops(phi, xx, T, nu1=1, nu2=1, m12=0, m21=0, gamma1=0, gamma2=0,
             h1=0.5, h2=0.5, theta0=1, initial_t=0, frozen1=False, 
             frozen2=False):
    """
    Integrate a 2-dimensional phi foward.

    phi: Initial 2-dimensional phi
    xx: 1-dimensional grid upon (0,1) overwhich phi is defined. It is assumed
        that this grid is used in all dimensions.

    nu's, gamma's, m's, and theta0 may be functions of time.
    nu1,nu2: Population sizes
    gamma1,gamma2: Selection coefficients on *all* segregating alleles
    h1,h2: Dominance coefficients. h = 0.5 corresponds to genic selection.
    m12,m21: Migration rates. Note that m12 is the rate *into 1 from 2*.
    theta0: Propotional to ancestral size. Typically constant.

    T: Time at which to halt integration
    initial_t: Time at which to start integration. (Note that this only matters
               if one of the demographic parameters is a function of time.)

    Note: Generalizing to different grids in different phi directions is
          straightforward. The tricky part will be later doing the extrapolation
          correctly.
    """
    phi = phi.copy()

    if T - initial_t == 0:
        return phi
    elif T - initial_t < 0:
        raise ValueError('Final integration time T (%f) is less than '
                         'intial_time (%f). Integration cannot be run '
                         'backwards.' % (T, initial_t))

    if (frozen1 or frozen2) and (m12 != 0 or m21 != 0):
        raise ValueError('Population cannot be frozen and have non-zero '
                         'migration to or from it.')

    vars_to_check = [nu1,nu2,m12,m21,gamma1,gamma2,h1,h2,theta0]
    if numpy.all([numpy.isscalar(var) for var in vars_to_check]):
        return _two_pops_const_params(phi, xx, T, nu1, nu2, m12, m21, 
                                      gamma1, gamma2, h1, h2, theta0, initial_t,
                                      frozen1, frozen2)
    yy = xx

    nu1_f = Misc.ensure_1arg_func(nu1)
    nu2_f = Misc.ensure_1arg_func(nu2)
    m12_f = Misc.ensure_1arg_func(m12)
    m21_f = Misc.ensure_1arg_func(m21)
    gamma1_f = Misc.ensure_1arg_func(gamma1)
    gamma2_f = Misc.ensure_1arg_func(gamma2)
    h1_f = Misc.ensure_1arg_func(h1)
    h2_f = Misc.ensure_1arg_func(h2)
    theta0_f = Misc.ensure_1arg_func(theta0)

    current_t = initial_t
    nu1,nu2 = nu1_f(current_t), nu2_f(current_t)
    m12,m21 = m12_f(current_t), m21_f(current_t)
    gamma1,gamma2 = gamma1_f(current_t), gamma2_f(current_t)
    h1,h2 = h1_f(current_t), h2_f(current_t)
    dx,dy = numpy.diff(xx),numpy.diff(yy)
    while current_t < T:
        dt = min(_compute_dt(dx,nu1,[m12],gamma1,h1),
                 _compute_dt(dy,nu2,[m21],gamma2,h2))
        this_dt = min(dt, T - current_t)

        next_t = current_t + this_dt

        nu1,nu2 = nu1_f(next_t), nu2_f(next_t)
        m12,m21 = m12_f(next_t), m21_f(next_t)
        gamma1,gamma2 = gamma1_f(next_t), gamma2_f(next_t)
        h1,h2 = h1_f(next_t), h2_f(next_t)
        theta0 = theta0_f(next_t)

        if numpy.any(numpy.less([T,nu1,nu2,m12,m21,theta0], 0)):
            raise ValueError('A time, population size, migration rate, or '
                             'theta0 is < 0. Has the model been mis-specified?')
        if numpy.any(numpy.equal([nu1,nu2], 0)):
            raise ValueError('A population size is 0. Has the model been '
                             'mis-specified?')

        _inject_mutations_2D(phi, this_dt, xx, yy, theta0, frozen1, frozen2)
        if not frozen1: 
            phi = int_c.implicit_2Dx(phi, xx, yy, nu1, m12, gamma1, h1,
                                     this_dt, use_delj_trick)
        if not frozen2: 
            phi = int_c.implicit_2Dy(phi, xx, yy, nu2, m21, gamma2, h2,
                                     this_dt, use_delj_trick)

        current_t = next_t
    return phi

def three_pops(phi, xx, T, nu1=1, nu2=1, nu3=1,
               m12=0, m13=0, m21=0, m23=0, m31=0, m32=0,
               gamma1=0, gamma2=0, gamma3=0, h1=0.5, h2=0.5, h3=0.5,
               theta0=1, initial_t=0, frozen1=False, frozen2=False,
               frozen3=False):
    """
    Integrate a 3-dimensional phi foward.

    phi: Initial 3-dimensional phi
    xx: 1-dimensional grid upon (0,1) overwhich phi is defined. It is assumed
        that this grid is used in all dimensions.

    nu's, gamma's, m's, and theta0 may be functions of time.
    nu1,nu2,nu3: Population sizes
    gamma1,gamma2,gamma3: Selection coefficients on *all* segregating alleles
    h1,h2,h3: Dominance coefficients. h = 0.5 corresponds to genic selection.
    m12,m13,m21,m23,m31,m32: Migration rates. Note that m12 is the rate 
                             *into 1 from 2*.
    theta0: Propotional to ancestral size. Typically constant.

    T: Time at which to halt integration
    initial_t: Time at which to start integration. (Note that this only matters
               if one of the demographic parameters is a function of time.)

    Note: Generalizing to different grids in different phi directions is
          straightforward. The tricky part will be later doing the extrapolation
          correctly.
    """
    phi = phi.copy()

    if T - initial_t == 0:
        return phi
    elif T - initial_t < 0:
        raise ValueError('Final integration time T (%f) is less than '
                         'intial_time (%f). Integration cannot be run '
                         'backwards.' % (T, initial_t))


    if (frozen1 and (m12 != 0 or m21 != 0 or m13 !=0 or m31 != 0))\
       or (frozen2 and (m12 != 0 or m21 != 0 or m23 !=0 or m32 != 0))\
       or (frozen3 and (m13 != 0 or m31 != 0 or m23 !=0 or m32 != 0)):
        raise ValueError('Population cannot be frozen and have non-zero '
                         'migration to or from it.')

    vars_to_check = [nu1,nu2,nu3,m12,m13,m21,m23,m31,m32,gamma1,gamma2,
                     gamma3,h1,h2,h3,theta0]
    if numpy.all([numpy.isscalar(var) for var in vars_to_check]):
        return _three_pops_const_params(phi, xx, T, nu1, nu2, nu3, 
                                        m12, m13, m21, m23, m31, m32, 
                                        gamma1, gamma2, gamma3, h1, h2, h3,
                                        theta0, initial_t)
    zz = yy = xx

    nu1_f = Misc.ensure_1arg_func(nu1)
    nu2_f = Misc.ensure_1arg_func(nu2)
    nu3_f = Misc.ensure_1arg_func(nu3)
    m12_f = Misc.ensure_1arg_func(m12)
    m13_f = Misc.ensure_1arg_func(m13)
    m21_f = Misc.ensure_1arg_func(m21)
    m23_f = Misc.ensure_1arg_func(m23)
    m31_f = Misc.ensure_1arg_func(m31)
    m32_f = Misc.ensure_1arg_func(m32)
    gamma1_f = Misc.ensure_1arg_func(gamma1)
    gamma2_f = Misc.ensure_1arg_func(gamma2)
    gamma3_f = Misc.ensure_1arg_func(gamma3)
    h1_f = Misc.ensure_1arg_func(h1)
    h2_f = Misc.ensure_1arg_func(h2)
    h3_f = Misc.ensure_1arg_func(h3)
    theta0_f = Misc.ensure_1arg_func(theta0)

    current_t = initial_t
    nu1,nu2,nu3 = nu1_f(current_t), nu2_f(current_t), nu3_f(current_t)
    m12,m13 = m12_f(current_t), m13_f(current_t)
    m21,m23 = m21_f(current_t), m23_f(current_t)
    m31,m32 = m31_f(current_t), m32_f(current_t)
    gamma1,gamma2 = gamma1_f(current_t), gamma2_f(current_t)
    gamma3 = gamma3_f(current_t)
    h1,h2,h3 = h1_f(current_t), h2_f(current_t), h3_f(current_t)
    dx,dy,dz = numpy.diff(xx),numpy.diff(yy),numpy.diff(zz)
    dt = min(_compute_dt(dx,nu1,[m12,m13],gamma1,h1),
             _compute_dt(dy,nu2,[m21,m23],gamma2,h2),
             _compute_dt(dz,nu3,[m31,m32],gamma3,h3))
    steps = 0
    while current_t < T:
        dt = min(_compute_dt(dx,nu1,[m12,m13],gamma1,h1),
                 _compute_dt(dy,nu2,[m21,m23],gamma2,h2),
                 _compute_dt(dz,nu3,[m31,m32],gamma3,h3))
        this_dt = min(dt, T - current_t)

        next_t = current_t + this_dt
        steps += 1

        nu1,nu2,nu3 = nu1_f(next_t), nu2_f(next_t), nu3_f(next_t)
        m12,m13 = m12_f(next_t), m13_f(next_t)
        m21,m23 = m21_f(next_t), m23_f(next_t)
        m31,m32 = m31_f(next_t), m32_f(next_t)
        gamma1,gamma2 = gamma1_f(next_t), gamma2_f(next_t)
        gamma3 = gamma3_f(next_t)
        h1,h2,h3 = h1_f(next_t), h2_f(next_t), h3_f(next_t)
        theta0 = theta0_f(next_t)

        if numpy.any(numpy.less([T,nu1,nu2,nu3,m12,m13,m21,m23,m31,m32,theta0],
                                0)):
            raise ValueError('A time, population size, migration rate, or '
                             'theta0 is < 0. Has the model been mis-specified?')
        if numpy.any(numpy.equal([nu1,nu2,nu3], 0)):
            raise ValueError('A population size is 0. Has the model been '
                             'mis-specified?')

        _inject_mutations_3D(phi, this_dt, xx, yy, zz, theta0,
                             frozen1, frozen2, frozen3)
        if not frozen1:
            phi = int_c.implicit_3Dx(phi, xx, yy, zz, nu1, m12, m13, 
                                     gamma1, h1, this_dt, use_delj_trick)
        if not frozen2:
            phi = int_c.implicit_3Dy(phi, xx, yy, zz, nu2, m21, m23, 
                                     gamma2, h2, this_dt, use_delj_trick)
        if not frozen3:
            phi = int_c.implicit_3Dz(phi, xx, yy, zz, nu3, m31, m32, 
                                     gamma3, h3, this_dt, use_delj_trick)

        current_t = next_t
    return phi

#
# Here are the python versions of the population genetic functions.
#
def _Vfunc(x, nu):
    return 1./nu * x*(1-x)
def _Mfunc1D(x, gamma, h):
    return gamma * 2*(h + (1-2*h)*x) * x*(1-x)
def _Mfunc2D(x,y, mxy, gamma, h):
    return mxy * (y-x) + gamma * 2*(h + (1-2*h)*x) * x*(1-x)
def _Mfunc3D(x,y,z, mxy,mxz, gamma, h):
    return mxy * (y-x) + mxz * (z-x) + gamma * 2*(h + (1-2*h)*x) * x*(1-x)

def _compute_dfactor(dx):
    r"""
    \Delta_j from the paper.
    """
    # Controls how we take the derivative of the flux. The values here depend
    #  on the fact that we're defining our probability integral using the
    #  trapezoid rule.
    dfactor = numpy.zeros(len(dx)+1)
    dfactor[1:-1] = 2/(dx[:-1] + dx[1:])
    dfactor[0] = 2/dx[0]
    dfactor[-1] = 2/dx[-1]
    return dfactor

def _compute_delj(dx, MInt, VInt, axis=0):
    r"""
    Chang an Cooper's \delta_j term. Typically we set this to 0.5.
    """
    # Chang and Cooper's fancy delta j trick...
    if use_delj_trick:
        # upslice will raise the dimensionality of dx and VInt to be appropriate
        # for functioning with MInt.
        upslice = [nuax for ii in range(MInt.ndim)]
        upslice [axis] = slice(None)

        wj = 2 *MInt*dx[upslice]
        epsj = numpy.exp(wj/VInt[upslice])
        delj = (-epsj*wj + epsj * VInt[upslice] - VInt[upslice])/(wj - epsj*wj)
        # These where statements filter out edge case for delj
        delj = numpy.where(numpy.isnan(delj), 0.5, delj)
        delj = numpy.where(numpy.isinf(delj), 0.5, delj)
    else:
        delj = 0.5
    return delj

def _one_pop_const_params(phi, xx, T, nu=1, gamma=0, h=0.5, theta0=1, 
                          initial_t=0):
    """
    Integrate one population with constant parameters.

    In this case, we can precompute our a,b,c matrices for the linear system
    we need to evolve. This we can efficiently do in Python, rather than 
    relying on C. The nice thing is that the Python is much faster to debug.
    """
    if numpy.any(numpy.less([T,nu,theta0], 0)):
        raise ValueError('A time, population size, migration rate, or theta0 '
                         'is < 0. Has the model been mis-specified?')
    if numpy.any(numpy.equal([nu], 0)):
        raise ValueError('A population size is 0. Has the model been '
                         'mis-specified?')

    M = _Mfunc1D(xx, gamma, h)
    MInt = _Mfunc1D((xx[:-1] + xx[1:])/2, gamma, h)
    V = _Vfunc(xx, nu)
    VInt = _Vfunc((xx[:-1] + xx[1:])/2, nu)

    dx = numpy.diff(xx)
    dfactor = _compute_dfactor(dx)
    delj = _compute_delj(dx, MInt, VInt)

    a = numpy.zeros(phi.shape)
    a[1:] += dfactor[1:]*(-MInt * delj - V[:-1]/(2*dx))

    c = numpy.zeros(phi.shape)
    c[:-1] += -dfactor[:-1]*(-MInt * (1-delj) + V[1:]/(2*dx))

    b = numpy.zeros(phi.shape)
    b[:-1] += -dfactor[:-1]*(-MInt * delj - V[:-1]/(2*dx))
    b[1:] += dfactor[1:]*(-MInt * (1-delj) + V[1:]/(2*dx))

    if(M[0] <= 0):
        b[0] += (0.5/nu - M[0])*2/dx[0]
    if(M[-1] >= 0):
        b[-1] += -(-0.5/nu - M[-1])*2/dx[-1]

    dt = _compute_dt(dx,nu,[0],gamma,h)
    current_t = initial_t
    while current_t < T:    
        this_dt = min(dt, T - current_t)

        _inject_mutations_1D(phi, this_dt, xx, theta0)
        r = phi/this_dt
        phi = tridiag.tridiag(a, b+1/this_dt, c, r)
        current_t += this_dt
    return phi

def _two_pops_const_params(phi, xx, T, nu1=1,nu2=1, m12=0, m21=0,
                           gamma1=0, gamma2=0, h1=0.5, h2=0.5, theta0=1, 
                           initial_t=0, frozen1=False, frozen2=False):
    """
    Integrate two populations with constant parameters.
    """
    if numpy.any(numpy.less([T,nu1,nu2,m12,m21,theta0], 0)):
        raise ValueError('A time, population size, migration rate, or theta0 '
                         'is < 0. Has the model been mis-specified?')
    if numpy.any(numpy.equal([nu1,nu2], 0)):
        raise ValueError('A population size is 0. Has the model been '
                         'mis-specified?')
    yy = xx

    # The use of nuax (= numpy.newaxis) here is for memory conservation. We
    # could just create big X and Y arrays which only varied along one axis,
    # but that would be wasteful.
    Vx = _Vfunc(xx, nu1)
    VxInt = _Vfunc((xx[:-1]+xx[1:])/2, nu1)
    Mx = _Mfunc2D(xx[:,nuax], yy[nuax,:], m12, gamma1, h1)
    MxInt = _Mfunc2D((xx[:-1,nuax]+xx[1:,nuax])/2, yy[nuax,:], m12, gamma1, h1)

    Vy = _Vfunc(yy, nu2)
    VyInt = _Vfunc((yy[1:]+yy[:-1])/2, nu2)
    My = _Mfunc2D(yy[nuax,:], xx[:,nuax], m21, gamma2, h2)
    MyInt = _Mfunc2D((yy[nuax,1:] + yy[nuax,:-1])/2, xx[:,nuax], m21, gamma2,h2)

    dx = numpy.diff(xx)
    dfact_x = _compute_dfactor(dx)
    deljx = _compute_delj(dx, MxInt, VxInt)

    dy = numpy.diff(yy)
    dfact_y = _compute_dfactor(dy)
    deljy = _compute_delj(dy, MyInt, VyInt, axis=1)

    # The nuax's here broadcast the our various arrays to have the proper shape
    # to fit into ax,bx,cx
    ax, bx, cx = [numpy.zeros(phi.shape) for ii in range(3)]
    ax[ 1:] += dfact_x[ 1:,nuax]*(-MxInt*deljx    - Vx[:-1,nuax]/(2*dx[:,nuax]))
    cx[:-1] += dfact_x[:-1,nuax]*( MxInt*(1-deljx)- Vx[ 1:,nuax]/(2*dx[:,nuax]))
    bx[:-1] += dfact_x[:-1,nuax]*( MxInt*deljx    + Vx[:-1,nuax]/(2*dx[:,nuax]))
    bx[ 1:] += dfact_x[ 1:,nuax]*(-MxInt*(1-deljx)+ Vx[ 1:,nuax]/(2*dx[:,nuax]))

    if Mx[0,0] <= 0:
        bx[0,0] += (0.5/nu1 - Mx[0,0])*2/dx[0]
    if Mx[-1,-1] >= 0:
        bx[-1,-1] += -(-0.5/nu1 - Mx[-1,-1])*2/dx[-1]

    ay, by, cy = [numpy.zeros(phi.shape) for ii in range(3)]
    ay[:, 1:] += dfact_y[ 1:]*(-MyInt*deljy     - Vy[nuax,:-1]/(2*dy))
    cy[:,:-1] += dfact_y[:-1]*( MyInt*(1-deljy) - Vy[nuax, 1:]/(2*dy))
    by[:,:-1] += dfact_y[:-1]*( MyInt*deljy     + Vy[nuax,:-1]/(2*dy))
    by[:, 1:] += dfact_y[ 1:]*(-MyInt*(1-deljy) + Vy[nuax, 1:]/(2*dy))

    if My[0,0] <= 0:
        by[0,0] += (0.5/nu2 - My[0,0])*2/dy[0]
    if My[-1,-1] >= 0:
        by[-1,-1] += -(-0.5/nu2 - My[-1,-1])*2/dy[-1]

    dt = min(_compute_dt(dx,nu1,[m12],gamma1,h1),
             _compute_dt(dy,nu2,[m21],gamma2,h2))
    current_t = initial_t
    while current_t < T:    
        this_dt = min(dt, T - current_t)
        _inject_mutations_2D(phi, this_dt, xx, yy, theta0, frozen1, frozen2)
        if not frozen1:
            phi = int_c.implicit_precalc_2Dx(phi, ax, bx, cx, this_dt)
        if not frozen2:
            phi = int_c.implicit_precalc_2Dy(phi, ay, by, cy, this_dt)
        current_t += this_dt

    return phi

def _three_pops_const_params(phi, xx, T, nu1=1, nu2=1, nu3=1, 
                             m12=0, m13=0, m21=0, m23=0, m31=0, m32=0, 
                             gamma1=0, gamma2=0, gamma3=0, 
                             h1=0.5, h2=0.5, h3=0.5, theta0=1, initial_t=0,
                             frozen1=False, frozen2=False, frozen3=False):
    """
    Integrate three population with constant parameters.
    """
    if numpy.any(numpy.less([T,nu1,nu2,nu3,m12,m13,m21,m23,m31,m32,theta0], 0)):
        raise ValueError('A time, population size, migration rate, or theta0 '
                         'is < 0. Has the model been mis-specified?')
    if numpy.any(numpy.equal([nu1,nu2,nu3], 0)):
        raise ValueError('A population size is 0. Has the model been '
                         'mis-specified?')
    zz = yy = xx

    Vx = _Vfunc(xx, nu1)
    VxInt = _Vfunc((xx[:-1]+xx[1:])/2, nu1)
    Mx = _Mfunc3D(xx[:,nuax,nuax], yy[nuax,:,nuax], zz[nuax,nuax,:], 
                 m12, m13, gamma1, h1)
    MxInt = _Mfunc3D((xx[:-1,nuax,nuax]+xx[1:,nuax,nuax])/2, yy[nuax,:,nuax], 
                    zz[nuax,nuax,:], m12, m13, gamma1, h1)

    dx = numpy.diff(xx)
    dfact_x = _compute_dfactor(dx)
    deljx = _compute_delj(dx, MxInt, VxInt)

    ax, bx, cx = [numpy.zeros(phi.shape) for ii in range(3)]
    ax[ 1:] += dfact_x[ 1:,nuax,nuax]*(-MxInt*deljx    
                                       - Vx[:-1,nuax,nuax]/(2*dx[:,nuax,nuax]))
    cx[:-1] += dfact_x[:-1,nuax,nuax]*( MxInt*(1-deljx)
                                       - Vx[ 1:,nuax,nuax]/(2*dx[:,nuax,nuax]))
    bx[:-1] += dfact_x[:-1,nuax,nuax]*( MxInt*deljx    
                                       + Vx[:-1,nuax,nuax]/(2*dx[:,nuax,nuax]))
    bx[ 1:] += dfact_x[ 1:,nuax,nuax]*(-MxInt*(1-deljx)
                                       + Vx[ 1:,nuax,nuax]/(2*dx[:,nuax,nuax]))

    if Mx[0,0,0] <= 0:
        bx[0,0,0] += (0.5/nu1 - Mx[0,0,0])*2/dx[0]
    if Mx[-1,-1,-1] >= 0:
        bx[-1,-1,-1] += -(-0.5/nu1 - Mx[-1,-1,-1])*2/dx[-1]

    # Memory consumption can be an issue in 3D, so we delete arrays after we're
    # done with them.
    del Vx,VxInt,Mx,MxInt,deljx

    Vy = _Vfunc(yy, nu2)
    VyInt = _Vfunc((yy[1:]+yy[:-1])/2, nu2)
    My = _Mfunc3D(yy[nuax,:,nuax], xx[:,nuax, nuax], zz[nuax,nuax,:],
                 m21, m23, gamma2, h2)
    MyInt = _Mfunc3D((yy[nuax,1:,nuax] + yy[nuax,:-1,nuax])/2, xx[:,nuax, nuax],
                    zz[nuax,nuax,:], m21, m23, gamma2, h2)

    dy = numpy.diff(yy)
    dfact_y = _compute_dfactor(dy)
    deljy = _compute_delj(dy, MyInt, VyInt, axis=1)

    ay, by, cy = [numpy.zeros(phi.shape) for ii in range(3)]
    ay[:, 1:] += dfact_y[nuax, 1:,nuax]*(-MyInt*deljy     
                                    - Vy[nuax,:-1,nuax]/(2*dy[nuax,:,nuax]))
    cy[:,:-1] += dfact_y[nuax,:-1,nuax]*( MyInt*(1-deljy) 
                                    - Vy[nuax, 1:,nuax]/(2*dy[nuax,:,nuax]))
    by[:,:-1] += dfact_y[nuax,:-1,nuax]*( MyInt*deljy     
                                    + Vy[nuax,:-1,nuax]/(2*dy[nuax,:,nuax]))
    by[:, 1:] += dfact_y[nuax, 1:,nuax]*(-MyInt*(1-deljy) 
                                    + Vy[nuax, 1:,nuax]/(2*dy[nuax,:,nuax]))

    if My[0,0,0] <= 0:
        by[0,0,0] += (0.5/nu2 - My[0,0,0])*2/dy[0]
    if My[-1,-1,-1] >= 0:
        by[-1,-1,-1] += -(-0.5/nu2 - My[-1,-1,-1])*2/dy[-1]

    del Vy,VyInt,My,MyInt,deljy

    Vz = _Vfunc(zz, nu3)
    VzInt = _Vfunc((zz[1:]+zz[:-1])/2, nu3)
    Mz = _Mfunc3D(zz[nuax,nuax,:], xx[:,nuax, nuax], yy[nuax,:,nuax],
                 m31, m32, gamma3, h3)
    MzInt = _Mfunc3D((zz[nuax,nuax,1:] + zz[nuax,nuax,:-1])/2, xx[:,nuax, nuax],
                    yy[nuax,:,nuax], m31, m32, gamma3, h3)

    dz = numpy.diff(zz)
    dfact_z = _compute_dfactor(dz)
    deljz = _compute_delj(dz, MzInt, VzInt, axis=2)

    az, bz, cz = [numpy.zeros(phi.shape) for ii in range(3)]
    az[:,:, 1:] += dfact_z[ 1:]*(-MzInt*deljz     - Vz[nuax,nuax,:-1]/(2*dz))
    cz[:,:,:-1] += dfact_z[:-1]*( MzInt*(1-deljz) - Vz[nuax,nuax, 1:]/(2*dz))
    bz[:,:,:-1] += dfact_z[:-1]*( MzInt*deljz     + Vz[nuax,nuax,:-1]/(2*dz))
    bz[:,:, 1:] += dfact_z[ 1:]*(-MzInt*(1-deljz) + Vz[nuax,nuax, 1:]/(2*dz))

    if Mz[0,0,0] <= 0:
        bz[0,0,0] += (0.5/nu3 - Mz[0,0,0])*2/dz[0]
    if Mz[-1,-1,-1] >= 0:
        bz[-1,-1,-1] += -(-0.5/nu3 - Mz[-1,-1,-1])*2/dz[-1]

    dt = min(_compute_dt(dx,nu1,[m12,m13],gamma1,h1),
             _compute_dt(dy,nu2,[m21,m23],gamma2,h2),
             _compute_dt(dz,nu3,[m31,m32],gamma3,h3))
    current_t = initial_t
    while current_t < T:    
        this_dt = min(dt, T - current_t)
        _inject_mutations_3D(phi, this_dt, xx, yy, zz, theta0,
                             frozen1, frozen2, frozen3)
        if not frozen1:
            phi = int_c.implicit_precalc_3Dx(phi, ax, bx, cx, this_dt)
        if not frozen2:
            phi = int_c.implicit_precalc_3Dy(phi, ay, by, cy, this_dt)
        if not frozen3:
            phi = int_c.implicit_precalc_3Dz(phi, az, bz, cz, this_dt)
        current_t += this_dt
    return phi
