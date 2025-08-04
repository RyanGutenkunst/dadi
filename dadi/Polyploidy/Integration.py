import dadi.Misc as Misc
import dadi.Demes as Demes
import numpy
from numpy import newaxis as nuax
import scipy.integrate
import dadi.tridiag_cython as tridiag
import dadi.Polyploidy.Int1D_poly as int1D
from enum import IntEnum

### ==========================================================================
### CONSTANTS + CODE FROM DIPLOID IMPLEMENTATION
### ==========================================================================

#: Controls use of GPUs and multiprocessing
cuda_enabled = False

#: Controls use of Chang and Cooper's delj trick, which seems to lower accuracy.
use_delj_trick = False

#: Controls timestep for integrations. This is a reasonable default for
#: gridsizes of ~60. See set_timescale_factor for better control.
timescale_factor = 1e-3

#: Whether to use old timestep method, which is old_timescale_factor * dx[0].
use_old_timestep = False
#: Factor for told timestep method.
old_timescale_factor = 0.1

# TODO ask Ryan about this
# I think this needs to be split up across ploidy types
# Now, the diploid selection params could be zeros and 
# compute dt would naively take in the auto params
# potential concern is that the timestep seems to affect the Richardson extrapolation
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

def _compute_dt_auto(dx, nu, ms, gam1, gam2, gam3, gam4):
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

    maxVM = max(0.125/nu, sum(ms),\
                2*max(0.25*(1-0.25)*numpy.abs(((((- 4*gam1  + 6*gam2 - 4*gam3 + gam4)*.25 +
                                                  (9*gam1 - 9*gam2 + 3*gam3)) * .25 +
                                                  (-6*gam1 + 3*gam2)) * .25 + 
                                                   gam1)),
                      0.5*(1-0.5)*numpy.abs(((((- 4*gam1  + 6*gam2 - 4*gam3 + gam4)*.5 +
                                                (9*gam1 - 9*gam2 + 3*gam3)) * .5 +
                                                (-6*gam1 + 3*gam2)) * .5 + 
                                                 gam1)),
                      0.75*(1-0.75)*numpy.abs(((((- 4*gam1  + 6*gam2 - 4*gam3 + gam4)*.75 +
                                                  (9*gam1 - 9*gam2 + 3*gam3)) * .75 +
                                                  (-6*gam1 + 3*gam2)) * .75 + 
                                                   gam1))
                    ))
    if maxVM > 0:
        dt = timescale_factor / maxVM
    else:
        dt = numpy.inf
    if dt == 0:
        raise ValueError('Timestep is zero. Values passed in are nu=%f, ms=%s,'
                         'gamma1=%f, gamma2=%f, gamma3=%f, gamma4=%f.' 
                         % (nu, str(ms), gam1, gam2, gam3, gam4))
    return dt

### ==========================================================================
### INJECT MUTATIONS FUNCTIONS FOR ALL PLOIDIES + DIMENSIONS
### ==========================================================================

# these are slightly restructured from Ryan's versions to be more compatible with 
# the ploidy arguments which specify which injection function to use
# can change these back if needed
def _inject_mutations_1D(dt, xx, theta0):
    """
    Inject novel mutations for a timestep.
    """
    new_mut = dt/xx[1] * theta0/2 * 2/(xx[2] - xx[0])
    return new_mut

def _inject_mutations_1D_auto(dt, xx, theta0):
    """
    Inject novel mutations for a timestep corrected for autotetraploids 
    under T = 2N generations time scaling.
    """
    new_mut = dt/xx[1] * theta0/4 * 2/(xx[2] - xx[0]) 
    return new_mut

### ==========================================================================
### CLASS DEFINITION FOR SPECIFYING PLOIDY
### ==========================================================================

# TODO think more about separating ALLO into ALLOA and ALLOB
# TODO for now this is fine, but if we add hexaploids, we'll need to specify auto as autotet
# TODO ask Ryan about this structure
class PloidyType(IntEnum):
    DIPLOID = 0
    AUTO = 1
    ALLO = 2
    
    def param_names(self):
        """Return parameter names for this ploidy type"""
        param_map = {
            PloidyType.DIPLOID: ['gamma', 'h'],
            PloidyType.AUTO: ['gamma1', 'gamma2', 'gamma3', 'gamma4'],
            PloidyType.ALLO: ['gamma01', 'gamma02', 'gamma10', 'gamma11', 
                              'gamma12', 'gamma20', 'gamma21', 'gamma22']
        }
        return param_map[self]
    
    def pack_sel_params(self, sel_dict, max_params=8):
        """Pack selection parameters into standardized array"""
        # Initialize with zeros
        sel_params = [0.0] * max_params
        
        if self == PloidyType.DIPLOID:
            sel_params[0] = sel_dict.get('gamma', 0)
            sel_params[1] = sel_dict.get('h', 0)
            
        elif self == PloidyType.AUTO:
                sel_params[0] = sel_dict.get('gamma1', 0)
                sel_params[1] = sel_dict.get('gamma2', 0)
                sel_params[2] = sel_dict.get('gamma3', 0)
                sel_params[3] = sel_dict.get('gamma4', 0)
                
        elif self == PloidyType.ALLO:
            param_names = self.param_names()
            for i, param_name in enumerate(param_names):
                sel_params[i] = sel_dict.get(param_name, 0)
        
        return sel_params

### ==========================================================================
### MAIN INTEGRATION FUNCTIONS
### ==========================================================================

# TODO remove bypass_const_params option - just for testing usage
def one_pop(phi, xx, T, sel_dict, ploidyflag=PloidyType.DIPLOID, nu=1, theta0=1.0, initial_t=0, 
            frozen=False, deme_ids=None, bypass_const_params=False):
    """
    Integrate a 1-dimensional phi foward.

    phi: Initial 1-dimensional phi
    xx: Grid upon (0,1) overwhich phi is defined.

    nu, gamma, and theta0 may be functions of time.
    nu: Population size
    gamma: Selection coefficient on *all* segregating alleles
    h: Dominance coefficient. h = 0.5 corresponds to genic selection. 
       Heterozygotes have fitness 1+2sh and homozygotes have fitness 1+2s.
    theta0: Propotional to ancestral size. Typically constant.
    beta: Breeding ratio, beta=Nf/Nm.

    T: Time at which to halt integration
    initial_t: Time at which to start integration. (Note that this only matters
               if one of the demographic parameters is a function of time.)

    ploidyflag: 0 for diploid, 1 for auto
    sel_dict: dictionary of selection parameters for given ploidy type

    frozen: If True, population is 'frozen' so that it does not change.
            In the one_pop case, this is equivalent to not running the
            integration at all.
    deme_ids: sequence of strings representing the names of demes
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
    
    # need this to be intc for compatibility with Cython integration
    ploidy = numpy.zeros(2, dtype=numpy.intc)
    ploidy[ploidyflag] = 1
    # get the selection variables
    sel = ploidyflag.pack_sel_params(sel_dict=sel_dict)
    vars_to_check = (nu, sel[0], sel[1], sel[2], sel[3], theta0)
    if not bypass_const_params:
        if numpy.all([numpy.isscalar(var) for var in vars_to_check]):
            Demes.cache.append(Demes.IntegrationConst(duration = T-initial_t, start_sizes = [nu], deme_ids=deme_ids))
            return _one_pop_const_params(phi, xx, T, sel[0], sel[1], sel[2], sel[3], ploidy,
                                      nu, theta0, initial_t)


    # The user will only pass in selection params/funcs for one type of pop
    # But, we can't anticipate which one they'll pass in
    # So, we define all possible params as 0 and then overwrite them as needed
    gamma = h = gam1 = gam2 = gam3 = gam4 = 0

    # overwrite the above params with the passed in params
    # (this allows me to come back to the class and add wrappers for other selection models)
    if ploidyflag == PloidyType.DIPLOID:
        gamma, h = sel[0], sel[1]
    elif ploidyflag == PloidyType.AUTO: 
        gam1, gam2, gam3, gam4 = sel[0], sel[1], sel[2], sel[3]    

    nu_f = Misc.ensure_1arg_func(nu)
    gamma_f = Misc.ensure_1arg_func(gamma)
    h_f = Misc.ensure_1arg_func(h)
    gam1_f = Misc.ensure_1arg_func(gam1)
    gam2_f = Misc.ensure_1arg_func(gam2)
    gam3_f = Misc.ensure_1arg_func(gam3)    
    gam4_f = Misc.ensure_1arg_func(gam4)
    theta0_f = Misc.ensure_1arg_func(theta0)

    current_t = initial_t
    nu = nu_f(current_t)
    gamma, h =  gamma_f(current_t), h_f(current_t)
    gam1, gam2, gam3, gam4 = gam1_f(current_t), gam2_f(current_t), gam3_f(current_t), gam4_f(current_t)

    dx = numpy.diff(xx)
    demes_hist = [[0, [nu], []]]
    while current_t < T:
        dt = ploidy[0] * _compute_dt(dx,nu,[0],gamma,h)
        dt_auto = ploidy[1] * _compute_dt_auto(dx,nu,[0],gam1,gam2,gam3,gam4)
        this_dt = min(max(dt, dt_auto), T - current_t)

        # Because this is an implicit method, I need the *next* time's params.
        # So there's a little inconsistency here, in that I'm estimating dt
        # using the last timepoints nu,gamma,h.
        next_t = current_t + this_dt
        nu = nu_f(next_t)
        gamma, h = gamma_f(next_t), h_f(next_t)
        gam1, gam2, gam3, gam4 = gam1_f(next_t), gam2_f(next_t), gam3_f(next_t), gam4_f(next_t)
        theta0 = theta0_f(next_t)
        # define the sel_vec for the current time to pass to the integrator
        # this is a little subtle, but essentially one set of sel params will always be
        # set to zero, so this is either [gamma, h, 0, 0] or [gam1, gam2, gam3, gam4]
        sel_vec = numpy.array([gamma+gam1, h+gam2, gam3, gam4])
       
        demes_hist.append([next_t, [nu], []])

        if numpy.any(numpy.less([T,nu,theta0], 0)):
            raise ValueError('A time, population size, migration rate, or '
                             'theta0 is < 0. Has the model been mis-specified?')
        if numpy.any(numpy.equal([nu], 0)):
            raise ValueError('A population size is 0. Has the model been '
                             'mis-specified?')

        phi[1] += ploidy[0]*_inject_mutations_1D(this_dt, xx, theta0)
        phi[1] += ploidy[1]*_inject_mutations_1D_auto(this_dt, xx, theta0)
        # Do each step in C, since it will be faster to compute the a,b,c
        # matrices there.
        phi = int1D.implicit_1Dx(phi, xx, nu, sel_vec, this_dt, 
                                 use_delj_trick=use_delj_trick, ploidy=ploidy)
        current_t = next_t
    Demes.cache.append(Demes.IntegrationNonConst(history = demes_hist, deme_ids=deme_ids))
    return phi

def one_pop_branching(phi, xx, T, sel_dict, ploidyflag=PloidyType.DIPLOID, nu=1, theta0=1.0, initial_t=0, 
            frozen=False, deme_ids=None, bypass_const_params=False):
    """
    Integrate a 1-dimensional phi foward.

    phi: Initial 1-dimensional phi
    xx: Grid upon (0,1) overwhich phi is defined.

    nu, gamma, and theta0 may be functions of time.
    nu: Population size
    gamma: Selection coefficient on *all* segregating alleles
    h: Dominance coefficient. h = 0.5 corresponds to genic selection. 
       Heterozygotes have fitness 1+2sh and homozygotes have fitness 1+2s.
    theta0: Propotional to ancestral size. Typically constant.
    beta: Breeding ratio, beta=Nf/Nm.

    T: Time at which to halt integration
    initial_t: Time at which to start integration. (Note that this only matters
               if one of the demographic parameters is a function of time.)

    ploidyflag: 0 for diploid, 1 for auto
    sel_dict: dictionary of selection parameters for given ploidy type

    frozen: If True, population is 'frozen' so that it does not change.
            In the one_pop case, this is equivalent to not running the
            integration at all.
    deme_ids: sequence of strings representing the names of demes
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

    
    ploidy = numpy.zeros(2, dtype=numpy.intc)
    ploidy[ploidyflag] = 1
    # get the selection variables
    sel = ploidyflag.pack_sel_params(sel_dict=sel_dict)
    vars_to_check = (nu, sel[0], sel[1], sel[2], sel[3], theta0)
    if not bypass_const_params:
        if numpy.all([numpy.isscalar(var) for var in vars_to_check]):
            Demes.cache.append(Demes.IntegrationConst(duration = T-initial_t, start_sizes = [nu], deme_ids=deme_ids))
            return _one_pop_const_params_branching(phi, xx, T, sel[0], sel[1], sel[2], sel[3], ploidy,
                                      nu, theta0, initial_t)


    # The user will only pass in selection params/funcs for one type of pop
    # But, we can't anticipate which one they'll pass in
    # So, we define all possible params as 0 and then overwrite them only as needed
    gamma = h = gam1 = gam2 = gam3 = gam4 = 0

    # overwrite the above params with the passed in params
    # (this allows me to come back to the class and add wrappers for other selection models)
    if ploidyflag == PloidyType.DIPLOID:
        gamma, h = sel[0], sel[1]
        gamma_f = Misc.ensure_1arg_func(gamma)
        h_f = Misc.ensure_1arg_func(h)
    else: 
        gam1, gam2, gam3, gam4 = sel[0], sel[1], sel[2], sel[3] 
        gam1_f = Misc.ensure_1arg_func(gam1)
        gam2_f = Misc.ensure_1arg_func(gam2)
        gam3_f = Misc.ensure_1arg_func(gam3)    
        gam4_f = Misc.ensure_1arg_func(gam4)   

    nu_f = Misc.ensure_1arg_func(nu)
    theta0_f = Misc.ensure_1arg_func(theta0)

    current_t = initial_t
    nu = nu_f(current_t)
    if ploidyflag == PloidyType.DIPLOID:
        gamma, h =  gamma_f(current_t), h_f(current_t)
    else:
        gam1, gam2, gam3, gam4 = gam1_f(current_t), gam2_f(current_t), gam3_f(current_t), gam4_f(current_t)

    dx = numpy.diff(xx)
    demes_hist = [[0, [nu], []]]
    while current_t < T:
        if ploidyflag == PloidyType.DIPLOID:
            dt = ploidy[0] * _compute_dt(dx,nu,[0],gamma,h)
        else:
            dt = ploidy[1] * _compute_dt_auto(dx,nu,[0],gam1,gam2,gam3,gam4)
        this_dt = min(dt, T - current_t)

        # Because this is an implicit method, I need the *next* time's params.
        # So there's a little inconsistency here, in that I'm estimating dt
        # using the last timepoints nu,gamma,h.
        next_t = current_t + this_dt
        if ploidyflag == PloidyType.DIPLOID:
            gamma, h = gamma_f(next_t), h_f(next_t)
        else:
            gam1, gam2, gam3, gam4 = gam1_f(next_t), gam2_f(next_t), gam3_f(next_t), gam4_f(next_t)
        nu = nu_f(next_t)
        theta0 = theta0_f(next_t)
        # define the sel_vec for the current time to pass to the integrator
        # this is a little subtle, but essentially one set of sel params will always be
        # set to zero, so this is either [gamma, h, 0, 0] or [gam1, gam2, gam3, gam4]
        sel_vec = numpy.array([gamma+gam1, h+gam2, gam3, gam4])
       
        demes_hist.append([next_t, [nu], []])

        if numpy.any(numpy.less([T,nu,theta0], 0)):
            raise ValueError('A time, population size, migration rate, or '
                             'theta0 is < 0. Has the model been mis-specified?')
        if numpy.any(numpy.equal([nu], 0)):
            raise ValueError('A population size is 0. Has the model been '
                             'mis-specified?')

        if ploidyflag == PloidyType.DIPLOID:
            phi[1] += _inject_mutations_1D(this_dt, xx, theta0)
        else:
            phi[1] += _inject_mutations_1D_auto(this_dt, xx, theta0)
        # Do each step in C, since it will be faster to compute the a,b,c
        # matrices there.
        phi = int1D.implicit_1Dx_branching(phi, xx, nu, sel_vec, this_dt, 
                                 use_delj_trick=use_delj_trick, ploidy=ploidy)
        current_t = next_t
    Demes.cache.append(Demes.IntegrationNonConst(history = demes_hist, deme_ids=deme_ids))
    return phi


# ============================================================================
# PYTHON FUNCTIONS AND CONST_PARAMS INTEGRATION
# ============================================================================

# Python versions of the popgen functions
def _Vfunc_auto(x, nu):
    return 1./(2.*nu) * x*(1-x) 
# use Horner's method here to evaluate the polynomial as an easy optimization
def _Mfunc1D_auto(x, gam1, gam2, gam3, gam4):
    poly = ((((-4*gam1 + 6*gam2 - 4*gam3 + gam4)*x +
            (9*gam1 - 9*gam2 + 3*gam3)) * x +
           (-6*gam1 + 3*gam2)) * x + 
           gam1)
    return x * (1 - x) * 2 * poly

def _Mfunc2D_auto(x, y, mxy, gam1, gam2, gam3, gam4):
    return mxy * (y-x) + x*(1-x) * 2*(gam1 + (- 6*gam1 + 3*gam2)*x 
                                      + (9*gam1 - 9*gam2 + 3*gam3 )*x**2 
                                      + (-4*gam1 + 6*gam2 - 4*gam3 + gam4)*x**3)

def _Vfunc(x, nu):
    return 1./nu * x*(1-x) 
def _Mfunc1D(x, gamma, h):
    return gamma * 2*(h + (1-2*h)*x) * x*(1-x)
def _Mfunc2D(x,y, mxy, gamma, h):
    return mxy * (y-x) + gamma * 2*(h + (1-2*h)*x) * x*(1-x)
def _Mfunc3D(x,y,z, mxy,mxz, gamma, h):
    return mxy * (y-x) + mxz * (z-x) + gamma * 2*(h + (1-2*h)*x) * x*(1-x)

# Python versions of grid spacing and del_j
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

        wj = 2 *MInt*dx[tuple(upslice)]
        epsj = numpy.exp(wj/VInt[tuple(upslice)])
        delj = (-epsj*wj + epsj * VInt[tuple(upslice)] - VInt[tuple(upslice)])/(wj - epsj*wj)
        # These where statements filter out edge case for delj
        delj = numpy.where(numpy.isnan(delj), 0.5, delj)
        delj = numpy.where(numpy.isinf(delj), 0.5, delj)
    else:
        delj = 0.5
    return delj


# Constant parameters, 1D integration
def _one_pop_const_params(phi, xx, T, sel0, sel1, sel2, sel3, ploidy, nu=1, theta0=1, 
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

    M = ploidy[0] * _Mfunc1D(xx, sel0, sel1)
    M += ploidy[1] * _Mfunc1D_auto(xx, sel0, sel1, sel2, sel3)
    
    MInt = ploidy[0] * _Mfunc1D((xx[:-1] + xx[1:])/2, sel0, sel1)
    MInt += ploidy[1] * _Mfunc1D_auto((xx[:-1] + xx[1:])/2, sel0, sel1, sel2, sel3)

    V = ploidy[0] * _Vfunc(xx, nu)
    V += ploidy[1] * _Vfunc_auto(xx, nu)

    VInt = ploidy[0] * _Vfunc((xx[:-1] + xx[1:])/2, nu)
    VInt += ploidy[1] * _Vfunc_auto((xx[:-1] + xx[1:])/2, nu)

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

    ### Here, the variance term is showing up again, but not explicitly
    ### So, we also need to add both BCs multiplied by the ploidy coeff
    if(M[0] <= 0):
        b[0] += ploidy[0]*(0.5/nu - M[0])*2/dx[0]
        b[0] += ploidy[1]*(0.25/nu - M[0])*2/dx[0]
    if(M[-1] >= 0):
        b[-1] += ploidy[0]*-(-0.5/nu - M[-1])*2/dx[-1]
        b[-1] += ploidy[1]*-(-0.25/nu - M[-1])*2/dx[-1]

    dt = ploidy[0] * _compute_dt(dx,nu,[0],sel0,sel1)
    dt_auto = ploidy[1] * _compute_dt_auto(dx,nu,[0],sel0,sel1,sel2,sel3)
    current_t = initial_t
    while current_t < T:    
        this_dt = min(max(dt, dt_auto), T - current_t)

        phi[1] += ploidy[0]*_inject_mutations_1D(this_dt, xx, theta0)
        phi[1] += ploidy[1]*_inject_mutations_1D_auto(this_dt, xx, theta0)
        r = phi/this_dt
        phi = tridiag.tridiag(a, b+1/this_dt, c, r)
        current_t += this_dt
    return phi

def _one_pop_const_params_branching(phi, xx, T, sel0, sel1, sel2, sel3, ploidy, nu=1, theta0=1, 
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

    dx = numpy.diff(xx)
    dfactor = _compute_dfactor(dx)

    if ploidy[0]:
        M = _Mfunc1D(xx, sel0, sel1)
        MInt = _Mfunc1D((xx[:-1] + xx[1:])/2, sel0, sel1)
        V = _Vfunc(xx, nu)
        VInt = _Vfunc((xx[:-1] + xx[1:])/2, nu)
        bc_factor = 0.5
        dt = _compute_dt(dx,nu,[0],sel0,sel1)
    else:
        M = _Mfunc1D_auto(xx, sel0, sel1, sel2, sel3)
        MInt = _Mfunc1D_auto((xx[:-1] + xx[1:])/2, sel0, sel1, sel2, sel3)
        V = _Vfunc_auto(xx, nu)
        VInt = _Vfunc_auto((xx[:-1] + xx[1:])/2, nu)
        bc_factor = 0.25
        dt = _compute_dt_auto(dx,nu,[0],sel0,sel1,sel2,sel3)

    delj = _compute_delj(dx, MInt, VInt)

    a = numpy.zeros(phi.shape)
    a[1:] += dfactor[1:]*(-MInt * delj - V[:-1]/(2*dx))

    c = numpy.zeros(phi.shape)
    c[:-1] += -dfactor[:-1]*(-MInt * (1-delj) + V[1:]/(2*dx))

    b = numpy.zeros(phi.shape)
    b[:-1] += -dfactor[:-1]*(-MInt * delj - V[:-1]/(2*dx))
    b[1:] += dfactor[1:]*(-MInt * (1-delj) + V[1:]/(2*dx))

    ### Here, the variance term is showing up again, but not explicitly
    ### So, we also need to adjust the BCs for ploidy
    if(M[0] <= 0):
        b[0] += (bc_factor/nu - M[0])*2/dx[0]
    if(M[-1] >= 0):
        b[-1] += -(-bc_factor/nu - M[-1])*2/dx[-1]

    current_t = initial_t
    while current_t < T:    
        this_dt = min(dt, T - current_t)

        if ploidy[0]:
            phi[1] += _inject_mutations_1D(this_dt, xx, theta0)
        else:
            phi[1] += _inject_mutations_1D_auto(this_dt, xx, theta0)
        r = phi/this_dt
        phi = tridiag.tridiag(a, b+1/this_dt, c, r)
        current_t += this_dt
    return phi
