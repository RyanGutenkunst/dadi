import dadi
import numpy
from numpy import newaxis as nuax
import dadi.tridiag_cython as tridiag
import dadi.integration_c as int_c

#: Controls use of GPUs and multiprocessing
cuda_enabled = False

#: Controls use of Chang and Cooper's delj trick, which seems to lower accuracy.
use_delj_trick = True

#: Controls timestep for integrations. This is a reasonable default for
#: gridsizes of ~60. See set_timescale_factor for better control.
timescale_factor = 1e-3

#: Whether to use old timestep method, which is old_timescale_factor * dx[0].
use_old_timestep = False
#: Factor for told timestep method.
old_timescale_factor = 0.1


def _inject_mutations_1D(phi, dt, xx, theta0):
    """
    Inject novel mutations for a timestep.
    """
    phi[1] += dt/xx[1] * theta0/2 * 2/(xx[2] - xx[0])
    return phi

def _inject_mutations_2D(phi, dt, xx, yy, theta0, frozen1, frozen2,
                         nomut1, nomut2):
    """
    Inject novel mutations for a timestep.
    """
    # Population 1
    if not frozen1 and not nomut1:
        phi[1,0] += dt/xx[1] * theta0/2 * 4/((xx[2] - xx[0]) * yy[1])
    # Population 2
    if not frozen2 and not nomut2:
        phi[0,1] += dt/yy[1] * theta0/2 * 4/((yy[2] - yy[0]) * xx[1])
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
def _compute_dt_auto(dx, nu, ms, gamma1, gamma2, gamma3, gamma4):
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

    ### I looked at this in Desmos for the equivalent function for autos and 
    ### the maximum value seems to sometimes be close to the 0.75 value 
    ### especially for recessive alleles, so I added that below

    # It might seem natural to scale dt based on dx[0]. However, testing has
    # shown that extrapolation is much more reliable when the same timesteps
    # are used in evaluations at different grid sizes.
    maxVM = max(0.25/nu, sum(ms),\
                2*max(0.25*(1-0.25)*numpy.abs(gamma1 - 6*.25*gamma1 + 3*.25*gamma2 + 9*.25**2*gamma1 
                                                - 9*.25**2*gamma2 - 4*.25**3*gamma1 + 3*.25**2*gamma3 
                                                + 6*.25**3*gamma2 - 4*.25**3*gamma3 + .25**3*gamma4),
                        0.5*(1-0.5)*numpy.abs(gamma1 - 6*.5*gamma1 + 3*.5*gamma2 + 9*.5**2*gamma1 
                                                - 9*.5**2*gamma2 - 4*.5**3*gamma1 + 3*.5**2*gamma3 
                                                + 6*.5**3*gamma2 - 4*.5**3*gamma3 + .5**3*gamma4),
                        0.75*(1-0.75)*numpy.abs(gamma1 - 6*.75*gamma1 + 3*.75*gamma2 + 9*.75**2*gamma1 
                                                - 9*.75**2*gamma2 - 4*.75**3*gamma1 + 3*.75**2*gamma3 
                                                + 6*.75**3*gamma2 - 4*.75**3*gamma3 + .75**3*gamma4)))
    if maxVM > 0:
        dt = timescale_factor / maxVM
    else:
        dt = numpy.inf
    if dt == 0:
        raise ValueError('Timestep is zero. Values passed in are nu=%f, ms=%s,'
                         'gamma1=%f, gamma2=%f, gamma3=%f, gamma4=%f.' 
                         % (nu, str(ms), gamma1, gamma2, gamma3, gamma4))
    return dt

#
# Here are the python versions of the population genetic functions.
#

### the variance is manually corrected here because the variance and difussion time
### should really be rescaled by 4N for autos instead of 2N as for diploids
def _Vfunc_auto(x, nu, beta=1):
    return 1./(2.*nu) * x*(1-x) * (beta+1.)**2/(4.*beta)
def _Vfunc_auto_prime(x, nu, beta=1):
    return 1./(2.*nu) * (1-2*x) * (beta+1.)**2/(4.*beta)
def _Mfunc1D_auto(x, gam1, gam2, gam3, gam4):
    return x*(1-x) * 2*(gam1 - 6*x*gam1 + 3*x*gam2 + 9*x**2*gam1 
                        - 9*x**2*gam2 - 4*x**3*gam1 + 3*x**2*gam3 
                        + 6*x**3*gam2 - 4*x**3*gam3 + x**3*gam4)
def _Mfunc2D_auto(x, y, mxy, gam1, gam2, gam3, gam4):
    return mxy * (y-x) + x*(1-x) * 2*(gam1 - 6*x*gam1 + 3*x*gam2 + 9*x**2*gam1 
                                    - 9*x**2*gam2 - 4*x**3*gam1 + 3*x**2*gam3 
                                    + 6*x**3*gam2 - 4*x**3*gam3 + x**3*gam4)

def _Vfunc(x, nu, beta=1):
    return 1./nu * x*(1-x) * (beta+1.)**2/(4.*beta)
def _Vfunc_prime(x, nu, beta=1):
    return 1./nu * (1-2*x) * (beta+1.)**2/(4.*beta)
def _Mfunc1D(x, gamma, h):
    return gamma * 2*(h + (1-2*h)*x) * x*(1-x)
def _Mfunc2D(x,y, mxy, gamma, h):
    return mxy * (y-x) + gamma * 2*(h + (1-2*h)*x) * x*(1-x)
def _Mfunc3D(x,y,z, mxy,mxz, gamma, h):
    return mxy * (y-x) + mxz * (z-x) + gamma * 2*(h + (1-2*h)*x) * x*(1-x)

### This shouldn't need to change because the trapezoid rule for integration is the same as for diploids 
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


### Two notes here
### 1. I am not sure how this is equivalent to the \delta_j from Chang and Cooper 1970
### 2. From testing, there seems to be some bug in the code that is fixed by changing [upslice] to [tuple(upslice)]
def _compute_delj(dx, MInt, VInt, VIntprime, axis=0):
    r"""
    Chang and Cooper's \delta_j term. Typically we set this to 0.5.
    """
    # Chang and Cooper's fancy delta j trick...
    if use_delj_trick:
        # upslice will raise the dimensionality of dx and VInt to be appropriate
        # for functioning with MInt.
        upslice = [nuax for ii in range(MInt.ndim)]
        upslice [axis] = slice(None)

        wj = ((VIntprime[tuple(upslice)]-2*MInt)*dx[tuple(upslice)])/VInt[tuple(upslice)]
        delj = numpy.where(
            numpy.abs(wj) < 1e-10,
            0.5,  # Pure diffusion limit
            (1.0/wj) - 1.0/(numpy.exp(wj) - 1.0)
        )
        # These where statements filter out edge case for delj
        delj = numpy.where(numpy.isnan(delj), 0.5, delj)
        delj = numpy.where(numpy.isinf(delj), 0.5, delj)
    else:
        delj = 0.5
    return delj


def _one_pop_const_params(phi, xx, T, nu=1, gamma1=0, gamma2=0, gamma3=0, gamma4=0, theta0=1, 
                          initial_t=0, beta=1):
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

    M = _Mfunc1D_auto(xx, gamma1, gamma2, gamma3, gamma4)
    MInt = _Mfunc1D_auto((xx[:-1] + xx[1:])/2, gamma1, gamma2, gamma3, gamma4)
    ### The V values here are already corrected for by the changes to _Vfunc_auto
    V = _Vfunc_auto(xx, nu, beta=beta)
    VInt = _Vfunc_auto((xx[:-1] + xx[1:])/2, nu, beta=beta)
    VIntprime = _Vfunc_auto_prime((xx[:-1] + xx[1:])/2, nu, beta=beta)

    dx = numpy.diff(xx)
    dfactor = _compute_dfactor(dx)
    delj = _compute_delj(dx, MInt, VInt, VIntprime)

    a = numpy.zeros(phi.shape)
    a[1:] += dfactor[1:]*(-MInt * delj - V[:-1]/(2*dx))

    c = numpy.zeros(phi.shape)
    c[:-1] += -dfactor[:-1]*(-MInt * (1-delj) + V[1:]/(2*dx))

    b = numpy.zeros(phi.shape)
    b[:-1] += -dfactor[:-1]*(-MInt * delj - V[:-1]/(2*dx))
    b[1:] += dfactor[1:]*(-MInt * (1-delj) + V[1:]/(2*dx))

    ### Here, the variance term is showing up again, but not explicitly
    ### Instead, this is a pre-evaluated derivative of V
    ### So, there is a 0.25/nu instead of a 0.5/nu in these BCs
    if(M[0] <= 0):
        b[0] += (0.25/nu - M[0])*2/dx[0]
    if(M[-1] >= 0):
        b[-1] += -(-0.25/nu - M[-1])*2/dx[-1]

    dt = _compute_dt_auto(dx,nu,[0],gamma1,gamma2,gamma3,gamma4)
    current_t = initial_t
    while current_t < T:    
        this_dt = min(dt, T - current_t)

        _inject_mutations_1D(phi, this_dt, xx, theta0)
        r = phi/this_dt
        phi = tridiag.tridiag(a, b+1/this_dt, c, r)
        current_t += this_dt
    return phi


# setting this up so that pop 1 is diploids and pop 2 is autotetraploids 
def _two_pops_const_params(phi, xx, T, nu1=1, nu2=1, m12=0, m21=0,
                           gamma1=0, h1=0.5, gamma2_1=0, gamma2_2=0, gamma2_3=0, 
                           gamma2_4=0, theta0=1, initial_t=0, frozen1=False, 
                           frozen2=False, nomut1=False, nomut2=False):
    """
    Integrate two populations (one diploid, the other autotetraploid) with constant parameters.
    """
    if numpy.any(numpy.less([T,nu1,nu2,m12,m21,theta0], 0)):
        raise ValueError('A time, population size, migration rate, or theta0 '
                         'is < 0. Has the model been mis-specified?')
    if numpy.any(numpy.equal([nu1,nu2], 0)):
        raise ValueError('A population size is 0. Has the model been '
                         'mis-specified?')
    # xx and Vx, Mx, etc. refer to the diploids which is pop 1 for the input parameters
    # yy and Vy, My, etc. refer to the autotetraploids which is pop 2
    
    yy = xx

    # The use of nuax (= numpy.newaxis) here is for memory conservation. We
    # could just create big X and Y arrays which only varied along one axis,
    # but that would be wasteful.
    # Diploid
    Vx = _Vfunc(xx, nu1)
    VxInt = _Vfunc((xx[:-1]+xx[1:])/2, nu1)
    VxIntprime = _Vfunc_prime((xx[:-1]+xx[1:])/2, nu1)
    Mx = _Mfunc2D(xx[:,nuax], yy[nuax,:], m12, gamma1, h1)
    MxInt = _Mfunc2D((xx[:-1,nuax]+xx[1:,nuax])/2, yy[nuax,:], m12, gamma1, h1)
    # Autotetraploids
    Vy = _Vfunc_auto(yy, nu2)
    VyInt = _Vfunc_auto((yy[1:]+yy[:-1])/2, nu2)
    VyIntprime = _Vfunc_auto_prime((yy[1:]+yy[:-1])/2, nu2)
    My = _Mfunc2D_auto(yy[nuax,:], xx[:,nuax], m21, gamma2_1, gamma2_2, gamma2_3, gamma2_4)
    MyInt = _Mfunc2D_auto((yy[nuax,1:] + yy[nuax,:-1])/2, xx[:,nuax], m21, gamma2_1, gamma2_2, gamma2_3, gamma2_4)

    dx = numpy.diff(xx)
    dfact_x = _compute_dfactor(dx)
    deljx = _compute_delj(dx, MxInt, VxInt, VxIntprime)

    dy = numpy.diff(yy)
    dfact_y = _compute_dfactor(dy)
    deljy = _compute_delj(dy, MyInt, VyInt, VyIntprime, axis=1)

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

    # Note the 0.25/nu2 factors here which correct for the derivative of the variance
    if My[0,0] <= 0:
        by[0,0] += (0.25/nu2 - My[0,0])*2/dy[0]
    if My[-1,-1] >= 0:
        by[-1,-1] += -(-0.25/nu2 - My[-1,-1])*2/dy[-1]

    # need to call both compute_dt and compute_dt_auto here
    dt = min(_compute_dt(dx,nu1,[m12],gamma1,h1),
             _compute_dt_auto(dy,nu2,[m21],gamma2_1,gamma2_2,gamma2_3,gamma2_4))
    current_t = initial_t

    ### comment out the cuda_enabled for now... this may be functionality to bring back in later
    # if cuda_enabled:
    #     import dadi.cuda
    #     phi = dadi.cuda.Integration._two_pops_const_params(phi, xx,
    #             theta0, frozen1, frozen2, nomut1, nomut2, ax, bx, cx, ay,
    #             by, cy, current_t, dt, T)
    #     return phi

    while current_t < T:
        this_dt = min(dt, T - current_t)
        _inject_mutations_2D(phi, this_dt, xx, yy, theta0, frozen1, frozen2,
                            nomut1, nomut2) 
        # The int_c functions essentially just wrap the tridiagonal method, so should be fine
        if not frozen1:
            phi = int_c.implicit_precalc_2Dx(phi, ax, bx, cx, this_dt)
        if not frozen2:
            phi = int_c.implicit_precalc_2Dy(phi, ay, by, cy, this_dt)
        current_t += this_dt

    return phi

