import dadi.Misc as Misc
import dadi.Polyploidy.MiscPoly as MiscPoly
import dadi.Demes as Demes
import numpy
from numpy import newaxis as nuax
import scipy.integrate
import dadi.tridiag_cython as tridiag
from . import Int1D_poly as int1D
from . import Int2D_poly as int2D
from . import Int3D_poly as int3D
from enum import IntEnum

### ==========================================================================
### CONSTANTS
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

### ==========================================================================
### COMPUTE DT FUNCTIONS
### ==========================================================================
def _compute_dt(dx, nu, ms, sel, ploidy):
    """
    Compute the timestep along a single dimension of phi. 

    Acts as a wrapper and calls _compute_dt_* for the corresponding ploidy type.

    sel: vector of selection parameters from unpacking the sel_dict
    ploidy: vector of length 4 of of ploidy coefficients
            e.g. [0, 1, 0, 0] specifies the current population as autotetraploid
            e.g. [0, 0, 0, 1] specifies the current population as allotetraploid subgenome b
    """
    if ploidy[0]:
        return _compute_dt_dip(dx, nu, ms, sel[0], sel[1])
    elif ploidy[1]:
        return _compute_dt_auto(dx, nu, ms, sel[0], sel[1], sel[2], sel[3])
    elif ploidy[2]:
        return _compute_dt_allo_a(dx, nu, ms, sel[0], sel[1], sel[2], sel[3], sel[4], sel[5], sel[6], sel[7])
    elif ploidy[3]:
        return _compute_dt_allo_b(dx, nu, ms, sel[0], sel[1], sel[2], sel[3], sel[4], sel[5], sel[6], sel[7])

def _compute_dt_dip(dx, nu, ms, gamma, h):
    """
    Compute the appropriate timestep given the current demographic params
    for diploids.

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
    Compute the appropriate timestep given the current demographic params
    for autotetraploids.

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

def _compute_dt_allo_a(dx, nu, ms, g01, g02, g10, g11, g12, g20, g21, g22):
    """
    Compute the appropriate timestep given the current demographic params
    for allotetraploid subgenome a.

    This is based on the maximum V or M expected in this direction. The
    timestep is scaled such that if the params are rescaled correctly by a
    constant, the exact same integration happens. (This is equivalent to
    multiplying the eqn through by some other 2N...)
    """
    if use_old_timestep:
        return old_timescale_factor * dx[0]

    # These are the maxima for V_func and M_func over the domain
    # It is difficult to know exactly where the maximum is, but for M_a it 
    # seems to be near x_a = 0.25, 0.5, 0.75 and x_b = 0, 1 
    # the nice thing is that x_b = 0, 1 are much simpler than the full M function
    
    # It might seem natural to scale dt based on dx[0]. However, testing has
    # shown that extrapolation is much more reliable when the same timesteps
    # are used in evaluations at different grid sizes.
    maxVM = max(0.25/nu, sum(ms),\
                2*max(0.25*(1-0.25)*numpy.abs(g10 + (-2*g10 + g20)*0.25), # x_a = 0.25, x_b = 0
                      0.5*(1-0.5)*numpy.abs(g10 + (-2*g10 + g20)*0.5), # x_a = 0.5, x_b = 0
                      0.75*(1-0.75)*numpy.abs(g10 + (-2*g10 + g20)*0.75), # x_a = 0.75, x_b = 0
                      0.25*(1-0.25)*numpy.abs(-g02 + g12 + (g02 -2*g12 + g22)*0.25), # x_a = 0.25, x_b = 1
                      0.5*(1-0.5)*numpy.abs(-g02 + g12 + (g02 -2*g12 + g22)*0.5), # x_a = 0.5, x_b = 1
                      0.75*(1-0.75)*numpy.abs(-g02 + g12 + (g02 -2*g12 + g22)*0.75))) # x_a = 0.75, x_b = 1
    if maxVM > 0:
        dt = timescale_factor / maxVM
    else:
        dt = numpy.inf
    if dt == 0:
        raise ValueError('Timestep is zero. Values passed in are nu=%f, ms=%s,'
                         'gamma01=%f, gamma02=%f, gamma10=%f, gamma11=%f, gamma12=%f,'
                         'gamma20=%f, gamma21=%f, gamma22=%f' 
                         % (nu, str(ms), g01, g02, g10, g11, g12, g20, g21, g22))
    return dt

def _compute_dt_allo_b(dx, nu, ms, g01, g02, g10, g11, g12, g20, g21, g22):
    """
    Compute the appropriate timestep given the current demographic params
    for allotetraploid subgenome b.

    This is based on the maximum V or M expected in this direction. The
    timestep is scaled such that if the params are rescaled correctly by a
    constant, the exact same integration happens. (This is equivalent to
    multiplying the eqn through by some other 2N...)
    """
    if use_old_timestep:
        return old_timescale_factor * dx[0]

    # These are the maxima for V_func and M_func over the domain
    # It is difficult to know exactly where the maximum is, but for M_a it 
    # seems to be near x_b = 0.25, 0.5, 0.75 and x_a = 0, 1 
    # the nice thing is that x_a = 0, 1 are much simpler than the full M function
    
    # It might seem natural to scale dt based on dx[0]. However, testing has
    # shown that extrapolation is much more reliable when the same timesteps
    # are used in evaluations at different grid sizes.
    maxVM = max(0.25/nu, sum(ms),\
                2*max(0.25*(1-0.25)*numpy.abs(g01 + (-2*g01 + g02)*0.25), # x_a = 0, x_b = 0.25
                      0.5*(1-0.5)*numpy.abs(g01 + (-2*g01 + g02)*0.5), # x_a = 0, x_b = 0.5
                      0.75*(1-0.75)*numpy.abs(g01 + (-2*g01 + g02)*0.75), # x_a = 0, x_b = 0.75
                      0.25*(1-0.25)*numpy.abs(-g20 + g21 + (g20 -2*g21 + g22)*0.25), # x_a = 1, x_b = 0.25
                      0.5*(1-0.5)*numpy.abs(-g20 + g21 + (g20 -2*g21 + g22)*0.5), # x_a = 1, x_b = 0.5
                      0.75*(1-0.75)*numpy.abs(-g20 + g21 + (g20 -2*g21 + g22)*0.75))) # x_a = 1, x_b = 0.75
    if maxVM > 0:
        dt = timescale_factor / maxVM
    else:
        dt = numpy.inf
    if dt == 0:
        raise ValueError('Timestep is zero. Values passed in are nu=%f, ms=%s,'
                         'gamma01=%f, gamma02=%f, gamma10=%f, gamma11=%f, gamma12=%f,'
                         'gamma20=%f, gamma21=%f, gamma22=%f' 
                         % (nu, str(ms), g01, g02, g10, g11, g12, g20, g21, g22))
    return dt

### ==========================================================================
### INJECT MUTATIONS FUNCTIONS FOR ALL PLOIDIES + DIMENSIONS
### ==========================================================================
# these are slightly restructured from Ryan's versions to be more compatible with 
# the ploidy arguments which specify which injection function to use
def _inject_mutations_1D(phi, dt, xx, theta0, ploidy):
    """
    Inject novel mutations for a timestep for diploids.
    """
    if ploidy[0]:
        phi[1] += dt/xx[1] * theta0/2 * 2/(xx[2] - xx[0])
    elif ploidy[1]:
        phi[1] += dt/xx[1] * theta0/2 * 4/((xx[2] - xx[0]) * xx[1])
    return phi

def _inject_mutations_2D(phi, dt, xx, yy, theta0, frozen1, frozen2,
                         nomut1, nomut2, ploidy1, ploidy2):
    """
    Inject novel mutations for a timestep.
    """
    if not frozen1 and not nomut1:
        if not ploidy1[1]: # this reads as if not autotetraploid
            phi[1,0] += dt/xx[1] * theta0/2 * 4/((xx[2] - xx[0]) * yy[1])
        else:
            phi[1,0] += dt/xx[1] * theta0/4 * 4/((xx[2] - xx[0]) * yy[1])
    if not frozen2 and not nomut2:
        if not ploidy2[1]:
            phi[0,1] += dt/yy[1] * theta0/2 * 4/((yy[2] - yy[0]) * xx[1])
        else:
            phi[0,1] += dt/yy[1] * theta0/4 * 4/((yy[2] - yy[0]) * xx[1])
    return phi

def _inject_mutations_3D(phi, dt, xx, yy, zz, theta0, frozen1, frozen2,
                         frozen3, ploidy1, ploidy2, ploidy3):
    """
    Inject novel mutations for a timestep.
    """
    if not frozen1:
        if not ploidy1[1]: # this reads as if not autotetraploid
            phi[1,0,0] += dt/xx[1] * theta0/2 * 8/((xx[2] - xx[0]) * yy[1] * zz[1])
        else:
            phi[1,0,0] += dt/xx[1] * theta0/4 * 8/((xx[2] - xx[0]) * yy[1] * zz[1])
    if not frozen2:
        if not ploidy2[1]:
            phi[0,1,0] += dt/yy[1] * theta0/2 * 8/((yy[2] - yy[0]) * xx[1] * zz[1])
        else:
            phi[0,1,0] += dt/yy[1] * theta0/4 * 8/((yy[2] - yy[0]) * xx[1] * zz[1])
    if not frozen3:
        if not ploidy3[1]:
            phi[0,0,1] += dt/zz[1] * theta0/2 * 8/((zz[2] - zz[0]) * xx[1] * yy[1])
        else:
            phi[0,0,1] += dt/zz[1] * theta0/4 * 8/((zz[2] - zz[0]) * xx[1] * yy[1])
    return phi

### ==========================================================================
### CLASS DEFINITION FOR SPECIFYING PLOIDY
### ==========================================================================
class PloidyType(IntEnum):
    DIPLOID = 0
    AUTO = 1
    ALLOa = 2
    ALLOb = 3
    
    def param_names(self):
        """Return parameter names for this ploidy type"""
        param_map = {
            PloidyType.DIPLOID: ['gamma', 'h'],
            PloidyType.AUTO: ['gamma1', 'gamma2', 'gamma3', 'gamma4'],
            PloidyType.ALLOa: ['gamma01', 'gamma02', 'gamma10', 'gamma11', 
                              'gamma12', 'gamma20', 'gamma21', 'gamma22'],
            PloidyType.ALLOb: ['gamma01', 'gamma02', 'gamma10', 'gamma11', 
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
                
        elif self == PloidyType.ALLOa:
            param_names = self.param_names()
            for i, param_name in enumerate(param_names):
                sel_params[i] = sel_dict.get(param_name, 0)
        
        elif self == PloidyType.ALLOb:
            param_names = self.param_names()
            for i, param_name in enumerate(param_names):
                sel_params[i] = sel_dict.get(param_name, 0)
        
        return sel_params

### ==========================================================================
### MAIN INTEGRATION FUNCTIONS FOR EACH DIMENSION
### ==========================================================================
def one_pop(phi, xx, T, nu=1, sel_dict = {'gamma':0, 'h':0.5}, ploidyflag=PloidyType.DIPLOID, theta0=1.0, initial_t=0, 
            frozen=False, deme_ids=None):
    """
    Integrate a 1-dimensional phi foward.

    phi: Initial 1-dimensional phi
    xx: Grid upon (0,1) overwhich phi is defined.

    nu, gamma, and theta0 may be functions of time.
    nu: Population size
    theta0: Propotional to ancestral size. Typically constant.

    T: Time at which to halt integration
    initial_t: Time at which to start integration. (Note that this only matters
               if one of the demographic parameters is a function of time.)

    sel_dict: dictionary of selection parameters for given ploidy type
    ploidyflag: See PloidyType class. 

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
    
    # vector of ploidy coefficients 
    # *only* for the 1 pop case is ploidy a vector of length 2
    # e.g. [0, 1] specifies the current population as autotetraploid
    # this is more convenient than calling if ploidyflag == PloidyType.xxx
    ploidy = numpy.zeros(2, dtype=numpy.intc)
    ploidy[ploidyflag] = 1
    # unpack the selection parameters from dict to list
    sel = ploidyflag.pack_sel_params(sel_dict)
    vars_to_check = [nu,sel,theta0]
    if numpy.all([numpy.isscalar(var) for var in vars_to_check]):
        Demes.cache.append(Demes.IntegrationConst(duration = T-initial_t, start_sizes = [nu], deme_ids=deme_ids))
        return _one_pop_const_params(phi, xx, T, sel, ploidy, nu, theta0, initial_t)

    # for convenience, we'll keep the sel_f as a vector of functions
    # this avoids explicitly writing all of the selection parameters for all of the ploidy types
    # and avoids many if statements which might slow things down
    sel_f = MiscPoly.ensure_1arg_func_vectorized(sel)
    nu_f = Misc.ensure_1arg_func(nu)
    theta0_f = Misc.ensure_1arg_func(theta0)

    current_t = initial_t
    nu = nu_f(current_t)
    sel = sel_f(current_t)

    dx = numpy.diff(xx)

    demes_hist = [[0, [nu], []]]
    while current_t < T:
        dt = _compute_dt(dx,nu,[0],sel,ploidy)
        this_dt = min(dt, T - current_t)

        # Because this is an implicit method, I need the *next* time's params.
        # So there's a little inconsistency here, in that I'm estimating dt
        # using the last timepoints nu,gamma,h.
        next_t = current_t + this_dt
        sel = sel_f(next_t)
        nu = nu_f(next_t)
        theta0 = theta0_f(next_t)
       
        demes_hist.append([next_t, [nu], []])

        if numpy.any(numpy.less([T,nu,theta0], 0)):
            raise ValueError('A time, population size, migration rate, or '
                             'theta0 is < 0. Has the model been mis-specified?')
        if numpy.any(numpy.equal([nu], 0)):
            raise ValueError('A population size is 0. Has the model been '
                             'mis-specified?')
        
        _inject_mutations_1D(phi, this_dt, xx, theta0, ploidy)
        # Do each step in C, since it will be faster to compute the a,b,c
        # matrices there.
        phi = int1D.implicit_1Dx(phi, xx, nu, sel, this_dt, 
                                 use_delj_trick, ploidy)
        current_t = next_t
    Demes.cache.append(Demes.IntegrationNonConst(history = demes_hist, deme_ids=deme_ids))
    return phi

def two_pops(phi, xx, T, nu1=1, nu2=1, m12=0, m21=0, sel_dict1 = {'gamma':0, 'h':0.5}, sel_dict2 = {'gamma':0, 'h':0.5},
            ploidyflag1=PloidyType.DIPLOID, ploidyflag2=PloidyType.DIPLOID, theta0=1, initial_t=0, frozen1=False,
             frozen2=False, nomut1=False, nomut2=False, enable_cuda_cached=False, deme_ids=None):
    """
    Integrate a 2-dimensional phi foward.

    phi: Initial 2-dimensional phi
    xx: 1-dimensional grid upon (0,1) overwhich phi is defined. It is assumed
        that this grid is used in all dimensions.

    nu's, gamma's, m's, h's, and theta0 may be functions of time.
    nu1,nu2: Population sizes
    m12,m21: Migration rates. Note that m12 is the rate *into 1 from 2*.
    theta0: Propotional to ancestral size. Typically constant.

    sel_dict1,2: dictionary of selection parameters for corresponding ploidy type of that population
    ploidyflag1,2: See PloidyType class. 
    
    T: Time at which to halt integration
    initial_t: Time at which to start integration. (Note that this only matters
               if one of the demographic parameters is a function of time.)

    frozen1,frozen2: If True, the corresponding population is "frozen" in time
                     (no new mutations and no drift), so the resulting spectrum
                     will correspond to an ancient DNA sample from that
                     population.

    nomut1,nomut2: If True, no new mutations will be introduced into the
                   given population.

    enable_cuda_cached: If True, enable CUDA integration with slower constant
                       parameter method. Likely useful only for benchmarking.
    deme_ids: sequence of strings representing the names of demes

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
    if cuda_enabled:
        raise ValueError('CUDA integration is not currently supported for polyploid models.')

    allo_types = {PloidyType.ALLOa, PloidyType.ALLOb}
    # check that at least one of the populations is an allo subgenome,
    # but the pair has not been specified as allo subgenomes of different types
    if ({ploidyflag1, ploidyflag2} in allo_types) and ({ploidyflag1, ploidyflag2} != allo_types):
        raise ValueError('Either population 1 and 2 is specified as allotetraploid subgenomes but not the other.'
                         'To model allotetraploids, the last two populations specified must be a pair of subgenomes.')

    if (ploidyflag1 in allo_types) or (ploidyflag2 in allo_types):
        if m12 != m21:
            raise ValueError('Population 1 or 2 is an allotetraploid subgenome. Both subgenomes must have the same migration rate.' 
                             'Here, the migration rates jointly specify a single exchange parameter and must be equal.'
                             'See Blischak et al. (2023) for details.')
        if nu1 != nu2:
            raise ValueError('Population 1 or 2 is an allotetraploid subgenome, but populations 1 and 2 do not have the same population size.'
                             'Has the model been mis-specified?')
        if sel_dict1 != sel_dict2:
            raise ValueError('Population 1 or 2 is an allotetraploid subgenome. Both populations must have the same selection parameters.')

    # create ploidy vectors with C integers
    ploidy1 = numpy.zeros(4, numpy.intc)
    ploidy2 = numpy.zeros(4, numpy.intc)
    ploidy1[ploidyflag1] = 1
    ploidy2[ploidyflag2] = 1

    # unpack selection params from dict to list
    sel1 = ploidyflag1.pack_sel_params(sel_dict1)
    sel2 = ploidyflag2.pack_sel_params(sel_dict2)

    # since sel1 and sel2 are lists, we need to unpack them using *
    vars_to_check = [nu1,nu2,m12,m21,*sel1,*sel2,theta0] 
    if numpy.all([numpy.isscalar(var) for var in vars_to_check]):
        # Constant integration with CUDA turns out to be slower,
        # so we only use it in specific circumsances.
        Demes.cache.append(Demes.IntegrationConst(duration = T-initial_t, 
                           start_sizes = [nu1, nu2], mig = [m12,m21], deme_ids=deme_ids))
        if not cuda_enabled or (cuda_enabled and enable_cuda_cached):
            return _two_pops_const_params(phi, xx, T, sel1, sel2, ploidy1, ploidy2, 
                                          nu1, nu2, m12, m21,theta0, initial_t,
                                          frozen1, frozen2, nomut1, nomut2)

    yy = xx

    sel1_f = MiscPoly.ensure_1arg_func_vectorized(sel1)
    sel2_f = MiscPoly.ensure_1arg_func_vectorized(sel2)
    nu1_f = Misc.ensure_1arg_func(nu1)
    nu2_f = Misc.ensure_1arg_func(nu2)
    m12_f = Misc.ensure_1arg_func(m12)
    m21_f = Misc.ensure_1arg_func(m21)
    theta0_f = Misc.ensure_1arg_func(theta0)

    # TODO: CUDA integration
    ### Ryan will need to implement this? 
    ### it is something I could take a look at... 
    # if cuda_enabled:
    #     import dadi.cuda
    #     phi = dadi.cuda.Integration._two_pops_temporal_params(phi, xx, T, initial_t,
    #             nu1_f, nu2_f, m12_f, m21_f, gamma1_f, gamma2_f, h1_f, h2_f, theta0_f, 
    #             frozen1, frozen2, nomut1, nomut2, deme_ids)
    #     return phi

    current_t = initial_t
    nu1,nu2 = nu1_f(current_t), nu2_f(current_t)
    m12,m21 = m12_f(current_t), m21_f(current_t)
    sel1, sel2 = sel1_f(current_t), sel2_f(current_t)
    
    dx,dy = numpy.diff(xx),numpy.diff(yy)

    demes_hist = [[0, [nu1,nu2], [m12,m21]]]
    while current_t < T:
        dt = min(_compute_dt(dx,nu1,[m12],sel1,ploidy1),
                 _compute_dt(dy,nu2,[m21],sel2,ploidy2))
        this_dt = min(dt, T - current_t)

        next_t = current_t + this_dt

        nu1,nu2 = nu1_f(next_t), nu2_f(next_t)
        m12,m21 = m12_f(next_t), m21_f(next_t)
        sel1, sel2 = sel1_f(next_t), sel2_f(next_t)
        theta0 = theta0_f(next_t)
        demes_hist.append([next_t, [nu1,nu2], [m12,m21]])

        if numpy.any(numpy.less([T,nu1,nu2,m12,m21,theta0], 0)):
            raise ValueError('A time, population size, migration rate, or '
                             'theta0 is < 0. Has the model been mis-specified?')
        if numpy.any(numpy.equal([nu1,nu2], 0)):
            raise ValueError('A population size is 0. Has the model been '
                             'mis-specified?')

        _inject_mutations_2D(phi, this_dt, xx, yy, theta0, frozen1, frozen2,
                             nomut1, nomut2, ploidy1, ploidy2)
        if not frozen1:
            phi = int2D.implicit_2Dx(phi, xx, yy, nu1, m12, sel1,
                                     this_dt, use_delj_trick, ploidy1)
        if not frozen2:
            phi = int2D.implicit_2Dy(phi, xx, yy, nu2, m21, sel2,
                                     this_dt, use_delj_trick, ploidy2)

        current_t = next_t
    Demes.cache.append(Demes.IntegrationNonConst(history = demes_hist, deme_ids=deme_ids))
    return phi

def three_pops(phi, xx, T, nu1=1, nu2=1, nu3=1,
               m12=0, m13=0, m21=0, m23=0, m31=0, m32=0,
               sel_dict1 = {'gamma':0, 'h':0.5}, sel_dict2 = {'gamma':0, 'h':0.5}, sel_dict3 = {'gamma':0, 'h':0.5},
               ploidyflag1=PloidyType.DIPLOID, ploidyflag2=PloidyType.DIPLOID, ploidyflag3=PloidyType.DIPLOID,
               theta0=1, initial_t=0, frozen1=False, frozen2=False,
               frozen3=False, enable_cuda_cached=False, deme_ids=None):
    """
    Integrate a 3-dimensional phi foward.

    phi: Initial 3-dimensional phi
    xx: 1-dimensional grid upon (0,1) overwhich phi is defined. It is assumed
        that this grid is used in all dimensions.

    nu's, gamma's, m's, and theta0 may be functions of time.
    nu1,nu2,nu3: Population sizes
    m12,m13,m21,m23,m31,m32: Migration rates. Note that m12 is the rate 
                             *into 1 from 2*.
    theta0: Propotional to ancestral size. Typically constant.

    sel_dict1,2,3: dictionary of selection parameters for corresponding ploidy type of that population
    ploidyflag1,2,3: See PloidyType class. 

    T: Time at which to halt integration
    initial_t: Time at which to start integration. (Note that this only matters
               if one of the demographic parameters is a function of time.)

    enable_cuda_cached: If True, enable CUDA integration with slower constant
                       parameter method. Likely useful only for benchmarking.
    deme_ids: sequence of strings representing the names of demes

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
    
    allo_types = {PloidyType.ALLOa, PloidyType.ALLOb}
    # check that at least one of the last two populations is an allo subgenome,
    # but the pair has not been specified as allo subgenomes of different types
    if ({ploidyflag2, ploidyflag3} in allo_types) and ({ploidyflag2, ploidyflag3} != allo_types):
        raise ValueError('Either population 1 and 2 is specified as allotetraploid subgenomes but not the other.'
                         'To model allotetraploids, the last two populations specified must be a pair of subgenomes.')

    if (ploidyflag2 in allo_types) or (ploidyflag3 in allo_types):
        if m23 != m32:
            raise ValueError('Population 2 or 3 is an allotetraploid subgenome. Both subgenomes must have the same migration rate.' 
                             'Here, the migration rates jointly specify a single exchange parameter and must be equal.'
                             'See Blischak et al. (2023) for details.')
        if nu1 != nu2:
            raise ValueError('Population 2 or 3 is an allotetraploid subgenome, but populations 2 and 3 do not have the same population size.'
                             'Has the model been mis-specified?')
        if sel_dict1 != sel_dict2:
            raise ValueError('Population 2 or 3 is an allotetraploid subgenome. Both populations must have the same selection parameters.')

    if ploidyflag1 in allo_types:
        raise ValueError('Population 1 is an allotetraploid subgenome.'  
                         'To model allotetraploids, the last two populations specified must be a pair of subgenomes.'
                         'The first population cannot be an allotetraploid subgenome.')

    # create ploidy vectors with C integers
    ploidy1 = numpy.zeros(4, numpy.intc)
    ploidy2 = numpy.zeros(4, numpy.intc)
    ploidy3 = numpy.zeros(4, numpy.intc)
    ploidy1[ploidyflag1] = 1
    ploidy2[ploidyflag2] = 1
    ploidy3[ploidyflag3] = 1

    # unpack selection params from dict to list
    sel1 = ploidyflag1.pack_sel_params(sel_dict1)
    sel2 = ploidyflag2.pack_sel_params(sel_dict2)
    sel3 = ploidyflag3.pack_sel_params(sel_dict3)

    # since sel1,2,3 are lists, we need to unpack them using *
    vars_to_check = [nu1,nu2,nu3,m12,m13,m21,m23,m31,m32,*sel1,*sel2,*sel3,theta0]
    if numpy.all([numpy.isscalar(var) for var in vars_to_check]):
        if not cuda_enabled or (cuda_enabled and enable_cuda_cached):
            Demes.cache.append(Demes.IntegrationConst(duration = T-initial_t, 
                               start_sizes = [nu1, nu2, nu3],
                               mig = [m12, m13, m21, m23, m31, m32], deme_ids=deme_ids))
            return _three_pops_const_params(phi, xx, T, 
                                            sel1, sel2, sel3,
                                            ploidy1, ploidy2, ploidy3,
                                            nu1, nu2, nu3,
                                            m12, m13, m21, m23, m31, m32,
                                            theta0, initial_t,
                                            frozen1, frozen2, frozen3)
    zz = yy = xx

    nu1_f, nu2_f, nu3_f = Misc.ensure_1arg_func(nu1), Misc.ensure_1arg_func(nu2), Misc.ensure_1arg_func(nu3)
    m12_f, m13_f = Misc.ensure_1arg_func(m12), Misc.ensure_1arg_func(m13)
    m21_f, m23_f = Misc.ensure_1arg_func(m21), Misc.ensure_1arg_func(m23)
    m31_f, m32_f = Misc.ensure_1arg_func(m31), Misc.ensure_1arg_func(m32)
    sel1_f, sel2_f, sel3_f = MiscPoly.ensure_1arg_func_vectorized(sel1), MiscPoly.ensure_1arg_func_vectorized(sel2), MiscPoly.ensure_1arg_func_vectorized(sel3)
    theta0_f = Misc.ensure_1arg_func(theta0)

    # TODO: CUDA integration
    # if cuda_enabled:
    #     import dadi.cuda
    #     phi = dadi.cuda.Integration._three_pops_temporal_params(phi, xx, T, initial_t,
    #             nu1_f, nu2_f, nu3_f, m12_f, m13_f, m21_f, m23_f, m31_f, m32_f, 
    #             gamma1_f, gamma2_f, gamma3_f, h1_f, h2_f, h3_f, 
    #             theta0_f, frozen1, frozen2, frozen3, deme_ids)
    #     return phi

    current_t = initial_t
    nu1,nu2,nu3 = nu1_f(current_t), nu2_f(current_t), nu3_f(current_t)
    m12,m13 = m12_f(current_t), m13_f(current_t)
    m21,m23 = m21_f(current_t), m23_f(current_t)
    m31,m32 = m31_f(current_t), m32_f(current_t)
    sel1, sel2, sel3 = sel1_f(current_t), sel2_f(current_t), sel3_f(current_t)
    
    dx,dy,dz = numpy.diff(xx),numpy.diff(yy),numpy.diff(zz)
    
    demes_hist = [[0, [nu1,nu2,nu3], [m12,m13,m21,m23,m31,m32]]]
    while current_t < T:
        dt = min(_compute_dt(dx,nu1,[m12,m13],sel1,ploidy1),
                 _compute_dt(dy,nu2,[m21,m23],sel2,ploidy2),
                 _compute_dt(dz,nu3,[m31,m32],sel3,ploidy3))
        this_dt = min(dt, T - current_t)

        next_t = current_t + this_dt

        nu1,nu2,nu3 = nu1_f(next_t), nu2_f(next_t), nu3_f(next_t)
        m12,m13 = m12_f(next_t), m13_f(next_t)
        m21,m23 = m21_f(next_t), m23_f(next_t)
        m31,m32 = m31_f(next_t), m32_f(next_t)
        sel1, sel2, sel3 = sel1_f(next_t), sel2_f(next_t), sel3_f(next_t)
        theta0 = theta0_f(next_t)
        demes_hist.append([next_t, [nu1,nu2,nu3], [m12,m13,m21,m23,m31,m32]])

        if numpy.any(numpy.less([T,nu1,nu2,nu3,m12,m13,m21,m23,m31,m32,theta0],
                                0)):
            raise ValueError('A time, population size, migration rate, or '
                             'theta0 is < 0. Has the model been mis-specified?')
        if numpy.any(numpy.equal([nu1,nu2,nu3], 0)):
            raise ValueError('A population size is 0. Has the model been '
                             'mis-specified?')

        _inject_mutations_3D(phi, this_dt, xx, yy, zz, theta0,
                             frozen1, frozen2, frozen3,
                             ploidy1, ploidy2, ploidy3)
        if not frozen1:
            phi = int3D.implicit_3Dx(phi, xx, yy, zz, nu1, m12, m13, 
                                     sel1, this_dt, use_delj_trick, ploidy1)
        if not frozen2:
            phi = int3D.implicit_3Dy(phi, xx, yy, zz, nu2, m21, m23, 
                                     sel2, this_dt, use_delj_trick, ploidy2)
        if not frozen3:
            phi = int3D.implicit_3Dz(phi, xx, yy, zz, nu3, m31, m32, 
                                     sel3, this_dt, use_delj_trick, ploidy3)

        current_t = next_t
    Demes.cache.append(Demes.IntegrationNonConst(history = demes_hist, deme_ids=deme_ids))
    return phi


# ============================================================================
# PYTHON FUNCTIONS AND CONST_PARAMS INTEGRATION
# ============================================================================
# Python versions of the popgen functions
# diploid
def _Vfunc(x, nu):
    return 1./nu * x*(1-x) 
def _Mfunc1D(x, gamma, h):
    return gamma * 2*(h + (1-2*h)*x) * x*(1-x)
def _Mfunc2D(x,y, mxy, gamma, h):
    return mxy * (y-x) + gamma * 2*(h + (1-2*h)*x) * x*(1-x)
def _Mfunc3D(x,y,z, mxy,mxz, gamma, h):
    return mxy * (y-x) + mxz * (z-x) + gamma * 2*(h + (1-2*h)*x) * x*(1-x)
# autotetraploid
def _Vfunc_auto(x, nu):
    return 1./nu * x*(1-x) / 2.
def _Mfunc1D_auto(x, gam1, gam2, gam3, gam4):
    poly = ((((-4*gam1 + 6*gam2 - 4*gam3 + gam4)*x +
            (9*gam1 - 9*gam2 + 3*gam3)) * x +
           (-6*gam1 + 3*gam2)) * x + 
           gam1)
    return x * (1 - x) * 2 * poly
def _Mfunc2D_auto(x, y, mxy, gam1, gam2, gam3, gam4):
    poly = ((((-4*gam1 + 6*gam2 - 4*gam3 + gam4)*x +
            (9*gam1 - 9*gam2 + 3*gam3)) * x +
           (-6*gam1 + 3*gam2)) * x + 
           gam1)
    return mxy * (y-x) + x*(1-x) * 2 * poly
def _Mfunc3D_auto(x, y, z, mxy, mxz, gam1, gam2, gam3, gam4):
    poly = ((((-4*gam1 + 6*gam2 - 4*gam3 + gam4)*x +
            (9*gam1 - 9*gam2 + 3*gam3)) * x +
           (-6*gam1 + 3*gam2)) * x + 
           gam1)
    return mxy * (y-x) + mxz * (z-x) + x*(1-x) * 2 * poly 
# allotetraploid
# here, g_ij refers to gamma_ij (not a gamete frequency!)
def _Mfunc2D_allo_a( x,  y,  mxy,  g01,  g02,  g10,  g11,  g12,  g20,  g21,  g22):
    # x is x_a, y is x_b
    xy = x*y
    yy = y*y
    xyy = xy*y
    poly = g10 + (-2*g10 + g20)*x + \
                  (-2*g01 - 2*g10 + 2*g11)*y + \
                  (2*g01 - g02 + g10 -2*g11 + g12)*yy + \
                  (-2*g01 + g02 - 2*g10 + 4*g11 -2*g12 + g20 -2*g21 + g22)*xyy + \
                  (2*g01 + 4*g10 -4*g11 -2*g20 +2*g21)*xy
    return mxy * (y-x) + x * (1. - x) * 2. * poly

def _Mfunc2D_allo_b( x,  y,  mxy,  g01,  g02,  g10,  g11,  g12,  g20,  g21,  g22):
    # x is x_b, y is x_a
    xy = x*y
    yy = y*y
    xyy = xy*y
    poly = g01 + (-2*g01 + g02)*x + \
                  (-2*g01 - 2*g10 + 2*g11)*y + \
                  (2*g10 - g20 + g01 -2*g11 + g21)*yy + \
                  (-2*g01 + g02 - 2*g10 + 4*g11 -2*g12 + g20 -2*g21 + g22)*xyy + \
                  (2*g10 + 4*g01 -4*g11 -2*g02 +2*g12)*xy
    return mxy * (y-x) + x * (1. - x) * 2. * poly

def _Mfunc3D_allo_a( x,  y, z,  mxy,  mxz,  g01,  g02,  g10,  g11,  g12,  g20,  g21,  g22):
    # x is x_a, y is x_b, z is a separate population
    xy = x*y
    yy = y*y
    xyy = xy*y
    poly = g10 + (-2*g10 + g20)*x + \
                  (-2*g01 - 2*g10 + 2*g11)*y + \
                  (2*g01 - g02 + g10 -2*g11 + g12)*yy + \
                  (-2*g01 + g02 - 2*g10 + 4*g11 -2*g12 + g20 -2*g21 + g22)*xyy + \
                  (2*g01 + 4*g10 -4*g11 -2*g20 +2*g21)*xy
    return mxy * (y-x) + mxz * (z-x) + x * (1. - x) * 2. * poly

def _Mfunc3D_allo_b( x,  y, z, mxy, mxz,  g01,  g02,  g10,  g11,  g12,  g20,  g21,  g22):
    # x is x_b, y is x_a, z is a separate population
    xy = x*y
    yy = y*y
    xyy = xy*y
    poly = g01 + (-2*g01 + g02)*x + \
                  (-2*g01 - 2*g10 + 2*g11)*y + \
                  (2*g10 - g20 + g01 -2*g11 + g21)*yy + \
                  (-2*g01 + g02 - 2*g10 + 4*g11 -2*g12 + g20 -2*g21 + g22)*xyy + \
                  (2*g10 + 4*g01 -4*g11 -2*g02 +2*g12)*xy
    return mxy * (y-x) + mxz * (z-x) + x * (1. - x) * 2. * poly

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
def _one_pop_const_params(phi, xx, T, s, ploidy, nu=1, theta0=1, 
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
        M = _Mfunc1D(xx, s[0], s[1])
        MInt = _Mfunc1D((xx[:-1] + xx[1:])/2, s[0], s[1])
        V = _Vfunc(xx, nu)
        VInt = _Vfunc((xx[:-1] + xx[1:])/2, nu)
        bc_factor = 0.5 # term for BCs, = (1-2x)/k eval. at x=0/x=1 for a k-ploid 
    else:
        M = _Mfunc1D_auto(xx, s[0], s[1], s[2], s[3])
        MInt = _Mfunc1D_auto((xx[:-1] + xx[1:])/2, s[0], s[1], s[2], s[3])
        V = _Vfunc_auto(xx, nu)
        VInt = _Vfunc_auto((xx[:-1] + xx[1:])/2, nu)
        bc_factor = 0.25 

    delj = _compute_delj(dx, MInt, VInt)

    a = numpy.zeros(phi.shape)
    a[1:] += dfactor[1:]*(-MInt * delj - V[:-1]/(2*dx))

    c = numpy.zeros(phi.shape)
    c[:-1] += -dfactor[:-1]*(-MInt * (1-delj) + V[1:]/(2*dx))

    b = numpy.zeros(phi.shape)
    b[:-1] += -dfactor[:-1]*(-MInt * delj - V[:-1]/(2*dx))
    b[1:] += dfactor[1:]*(-MInt * (1-delj) + V[1:]/(2*dx))

    if(M[0] <= 0):
        b[0] += (bc_factor/nu - M[0])*2/dx[0]
    if(M[-1] >= 0):
        b[-1] += -(-bc_factor/nu - M[-1])*2/dx[-1]

    dt = _compute_dt(dx,nu,[0],s,ploidy)
    current_t = initial_t
    while current_t < T:    
        this_dt = min(dt, T - current_t)

        _inject_mutations_1D(phi, dt, xx, theta0, ploidy)
        r = phi/this_dt
        phi = tridiag.tridiag(a, b+1/this_dt, c, r)
        current_t += this_dt
    return phi

def _two_pops_const_params(phi, xx, T, s1, s2, ploidy1, ploidy2, nu1=1,nu2=1, m12=0, m21=0,
                           theta0=1, initial_t=0, frozen1=False, frozen2=False,
                           nomut1=False, nomut2=False):
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

    # implicit in the x direction
    dx = numpy.diff(xx)
    dfact_x = _compute_dfactor(dx)

    if ploidy1[0]:
        Vx = _Vfunc(xx, nu1)
        VxInt = _Vfunc((xx[:-1]+xx[1:])/2, nu1)
        Mx = _Mfunc2D(xx[:,nuax], yy[nuax,:], m12, s1[0], s1[1])
        MxInt = _Mfunc2D((xx[:-1,nuax]+xx[1:,nuax])/2, yy[nuax,:], m12, s1[0], s1[1])
        deljx = _compute_delj(dx, MxInt, VxInt)
        bc_factorx = 0.5 
    elif ploidy1[1]:
        Vx = _Vfunc_auto(xx, nu1)
        VxInt = _Vfunc_auto((xx[:-1]+xx[1:])/2, nu1)
        Mx = _Mfunc2D_auto(xx[:,nuax], yy[nuax,:], m12, s1[0],s1[1],s1[2],s1[3])
        MxInt = _Mfunc2D_auto((xx[:-1,nuax]+xx[1:,nuax])/2, yy[nuax,:], m12, s1[0],s1[1],s1[2],s1[3])
        deljx = _compute_delj(dx, MxInt, VxInt)
        bc_factorx = 0.25 
    elif ploidy1[2]:
        Vx = _Vfunc(xx, nu1)
        VxInt = _Vfunc((xx[:-1]+xx[1:])/2, nu1)
        Mx = _Mfunc2D_allo_a(xx[:,nuax], yy[nuax,:], m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
        MxInt = _Mfunc2D_allo_a((xx[:-1,nuax]+xx[1:,nuax])/2, yy[nuax,:], m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
        deljx = _compute_delj(dx, MxInt, VxInt)
        bc_factorx = 0.5 
    elif ploidy1[3]:
        Vx = _Vfunc(xx, nu1)
        VxInt = _Vfunc((xx[:-1]+xx[1:])/2, nu1)
        Mx = _Mfunc2D_allo_b(xx[:,nuax], yy[nuax,:], m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
        MxInt = _Mfunc2D_allo_b((xx[:-1,nuax]+xx[1:,nuax])/2, yy[nuax,:], m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
        deljx = _compute_delj(dx, MxInt, VxInt)
        bc_factorx = 0.5 
    # The nuax's here broadcast the our various arrays to have the proper shape
    # to fit into ax,bx,cx
    ax, bx, cx = [numpy.zeros(phi.shape) for ii in range(3)]
    ax[ 1:] += dfact_x[ 1:,nuax]*(-MxInt*deljx    - Vx[:-1,nuax]/(2*dx[:,nuax]))
    cx[:-1] += dfact_x[:-1,nuax]*( MxInt*(1-deljx)- Vx[ 1:,nuax]/(2*dx[:,nuax]))
    bx[:-1] += dfact_x[:-1,nuax]*( MxInt*deljx    + Vx[:-1,nuax]/(2*dx[:,nuax]))
    bx[ 1:] += dfact_x[ 1:,nuax]*(-MxInt*(1-deljx)+ Vx[ 1:,nuax]/(2*dx[:,nuax]))

    if Mx[0,0] <= 0:
        bx[0,0] += (bc_factorx/nu1 - Mx[0,0])*2/dx[0]
    if Mx[-1,-1] >= 0:
        bx[-1,-1] += -(-bc_factorx/nu1 - Mx[-1,-1])*2/dx[-1]

    # implicit in the y direction
    dy = numpy.diff(yy)
    dfact_y = _compute_dfactor(dy)

    if ploidy2[0]:
        Vy = _Vfunc(yy, nu2)
        VyInt = _Vfunc((yy[1:]+yy[:-1])/2, nu2)
        My = _Mfunc2D(yy[nuax,:], xx[:,nuax], m21, s2[0],s2[1])
        MyInt = _Mfunc2D((yy[nuax,1:] + yy[nuax,:-1])/2, xx[:,nuax], m21, s2[0],s2[1])
        deljy = _compute_delj(dy, MyInt, VyInt, axis=1)
        bc_factory = 0.5 
    elif ploidy2[1]:
        Vy = _Vfunc_auto(yy, nu2)
        VyInt = _Vfunc_auto((yy[1:]+yy[:-1])/2, nu2)
        My = _Mfunc2D_auto(yy[nuax,:], xx[:,nuax], m21, s2[0],s2[1],s2[2],s2[3])
        MyInt = _Mfunc2D_auto((yy[nuax,1:] + yy[nuax,:-1])/2, xx[:,nuax], m21, s2[0],s2[1],s2[2],s2[3])
        deljy = _compute_delj(dy, MyInt, VyInt, axis=1)
        bc_factory = 0.25 
    elif ploidy2[2]:
        Vy = _Vfunc(yy, nu2)
        VyInt = _Vfunc((yy[1:]+yy[:-1])/2, nu2)
        My = _Mfunc2D_allo_a(yy[nuax,:], xx[:,nuax], m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
        MyInt = _Mfunc2D_allo_a((yy[nuax,1:] + yy[nuax,:-1])/2, xx[:,nuax], m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
        deljy = _compute_delj(dy, MyInt, VyInt, axis=1)
        bc_factory = 0.5 
    elif ploidy2[3]:
        Vy = _Vfunc(yy, nu2)
        VyInt = _Vfunc((yy[1:]+yy[:-1])/2, nu2)
        My = _Mfunc2D_allo_b(yy[nuax,:], xx[:,nuax], m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
        MyInt = _Mfunc2D_allo_b((yy[nuax,1:] + yy[nuax,:-1])/2, xx[:,nuax], m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
        deljy = _compute_delj(dy, MyInt, VyInt, axis=1)
        bc_factory = 0.5 
    # The nuax's here broadcast the our various arrays to have the proper shape
    # to fit into ax,bx,cx
    ay, by, cy = [numpy.zeros(phi.shape) for ii in range(3)]
    ay[:, 1:] += dfact_y[ 1:]*(-MyInt*deljy     - Vy[nuax,:-1]/(2*dy))
    cy[:,:-1] += dfact_y[:-1]*( MyInt*(1-deljy) - Vy[nuax, 1:]/(2*dy))
    by[:,:-1] += dfact_y[:-1]*( MyInt*deljy     + Vy[nuax,:-1]/(2*dy))
    by[:, 1:] += dfact_y[ 1:]*(-MyInt*(1-deljy) + Vy[nuax, 1:]/(2*dy))

    if My[0,0] <= 0:
        by[0,0] += (bc_factory/nu2 - My[0,0])*2/dy[0]
    if My[-1,-1] >= 0:
        by[-1,-1] += -(-bc_factory/nu2 - My[-1,-1])*2/dy[-1]

    dt = min(_compute_dt(dx,nu1,[m12],s1,ploidy1),
             _compute_dt(dy,nu2,[m21],s2,ploidy2))
    current_t = initial_t
    # TODO: CUDA integration
    if cuda_enabled:
        import dadi.cuda
        phi = dadi.cuda.Integration._two_pops_const_params(phi, xx,
                theta0, frozen1, frozen2, nomut1, nomut2, ax, bx, cx, ay,
                by, cy, current_t, dt, T)
        return phi

    while current_t < T:
        this_dt = min(dt, T - current_t)
        _inject_mutations_2D(phi, this_dt, xx, yy, theta0, frozen1, frozen2,
                            nomut1, nomut2, ploidy1, ploidy2)
        if not frozen1:
            phi = int2D.implicit_precalc_2Dx(phi, ax, bx, cx, this_dt)
        if not frozen2:
            phi = int2D.implicit_precalc_2Dy(phi, ay, by, cy, this_dt)
        current_t += this_dt

    return phi

def _three_pops_const_params(phi, xx, T, s1, s2, s3, ploidy1, ploidy2, ploidy3, 
                             nu1=1, nu2=1, nu3=1, 
                             m12=0, m13=0, m21=0, m23=0, m31=0, m32=0, 
                             theta0=1, initial_t=0,
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

    dx = numpy.diff(xx)
    dfact_x = _compute_dfactor(dx)

    if ploidy1[0]:
        Vx = _Vfunc(xx, nu1)
        VxInt = _Vfunc((xx[:-1]+xx[1:])/2, nu1)
        Mx = _Mfunc3D(xx[:,nuax,nuax], yy[nuax,:,nuax], zz[nuax,nuax,:], 
                      m12, m13, s1[0], s1[1])
        MxInt = _Mfunc3D((xx[:-1,nuax,nuax]+xx[1:,nuax,nuax])/2, yy[nuax,:,nuax], 
                          zz[nuax,nuax,:], m12, m13, s1[0], s1[1])
        deljx = _compute_delj(dx, MxInt, VxInt)
        bc_factorx = 0.5
    if ploidy1[1]:
        Vx = _Vfunc_auto(xx, nu1)
        VxInt = _Vfunc_auto((xx[:-1]+xx[1:])/2, nu1)
        Mx = _Mfunc3D_auto(xx[:,nuax,nuax], yy[nuax,:,nuax], zz[nuax,nuax,:], 
                      m12, m13, s1[0],s1[1],s1[2],s1[3])
        MxInt = _Mfunc3D_auto((xx[:-1,nuax,nuax]+xx[1:,nuax,nuax])/2, yy[nuax,:,nuax], 
                          zz[nuax,nuax,:], m12, m13, s1[0],s1[1],s1[2],s1[3])
        deljx = _compute_delj(dx, MxInt, VxInt)
        bc_factorx = 0.25 
    if ploidy1[2]:
        Vx = _Vfunc(xx, nu1)
        VxInt = _Vfunc((xx[:-1]+xx[1:])/2, nu1)
        Mx = _Mfunc3D_allo_a(xx[:,nuax,nuax], yy[nuax,:,nuax], zz[nuax,nuax,:], 
                      m12, m13, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
        MxInt = _Mfunc3D_allo_a((xx[:-1,nuax,nuax]+xx[1:,nuax,nuax])/2, yy[nuax,:,nuax], 
                          zz[nuax,nuax,:], m12, m13, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
        deljx = _compute_delj(dx, MxInt, VxInt)
        bc_factorx = 0.5 
    if ploidy1[3]:
        Vx = _Vfunc(xx, nu1)
        VxInt = _Vfunc((xx[:-1]+xx[1:])/2, nu1)
        Mx = _Mfunc3D_allo_b(xx[:,nuax,nuax], yy[nuax,:,nuax], zz[nuax,nuax,:], 
                      m12, m13, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
        MxInt = _Mfunc3D_allo_b((xx[:-1,nuax,nuax]+xx[1:,nuax,nuax])/2, yy[nuax,:,nuax], 
                          zz[nuax,nuax,:], m12, m13, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
        deljx = _compute_delj(dx, MxInt, VxInt)
        bc_factorx = 0.5 


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
        bx[0,0,0] += (bc_factorx/nu1 - Mx[0,0,0])*2/dx[0]
    if Mx[-1,-1,-1] >= 0:
        bx[-1,-1,-1] += -(-bc_factorx/nu1 - Mx[-1,-1,-1])*2/dx[-1]

    # Memory consumption can be an issue in 3D, so we delete arrays after we're
    # done with them.
    del Vx,VxInt,Mx,MxInt,deljx

    dy = numpy.diff(yy)
    dfact_y = _compute_dfactor(dy)
    if ploidy2[0]:
        Vy = _Vfunc(yy, nu2)
        VyInt = _Vfunc((yy[1:]+yy[:-1])/2, nu2)
        # note that the order of the params passed to _Mfunc3D for y and z is different from 
        # Ryan's original code. This is for consistency with the allo cases where
        # the first two dimensions passed to _Mfunc need to be the allo subgenomes 
        # and the subgenomes are always passed to the integrator as y and z.
        My = _Mfunc3D(yy[nuax,:,nuax], zz[nuax,nuax,:], xx[:,nuax, nuax],
                      m23, m21, s2[0], s2[1])
        MyInt = _Mfunc3D((yy[nuax,1:,nuax] + yy[nuax,:-1,nuax])/2, zz[nuax,nuax,:], 
                          xx[:,nuax, nuax], m23, m21, s2[0], s2[1])
        deljy = _compute_delj(dy, MyInt, VyInt, axis=1)
        bc_factory = 0.5
    if ploidy2[1]:
        Vy = _Vfunc_auto(yy, nu2)
        VyInt = _Vfunc_auto((yy[1:]+yy[:-1])/2, nu2)
        My = _Mfunc3D_auto(yy[nuax,:,nuax], zz[nuax,nuax,:], xx[:,nuax, nuax],
                      m23, m21, s2[0],s2[1],s2[2],s2[3])
        MyInt = _Mfunc3D_auto((yy[nuax,1:,nuax] + yy[nuax,:-1,nuax])/2, zz[nuax,nuax,:], 
                          xx[:,nuax, nuax], m23, m21, s2[0],s2[1],s2[2],s2[3])
        deljy = _compute_delj(dy, MyInt, VyInt, axis=1)
        bc_factory = 0.25
    if ploidy2[2]:
        Vy = _Vfunc(yy, nu2)
        VyInt = _Vfunc((yy[1:]+yy[:-1])/2, nu2)
        My = _Mfunc3D_allo_a(yy[nuax,:,nuax], zz[nuax,nuax,:], xx[:,nuax, nuax],
                      m23, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
        MyInt = _Mfunc3D_allo_a((yy[nuax,1:,nuax] + yy[nuax,:-1,nuax])/2, zz[nuax,nuax,:], 
                          xx[:,nuax, nuax], m23, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
        deljy = _compute_delj(dy, MyInt, VyInt, axis=1)
        bc_factory = 0.5
    if ploidy2[3]:
        Vy = _Vfunc(yy, nu2)
        VyInt = _Vfunc((yy[1:]+yy[:-1])/2, nu2)
        My = _Mfunc3D_allo_b(yy[nuax,:,nuax], zz[nuax,nuax,:], xx[:,nuax, nuax],
                      m23, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
        MyInt = _Mfunc3D_allo_b((yy[nuax,1:,nuax] + yy[nuax,:-1,nuax])/2, zz[nuax,nuax,:], 
                          xx[:,nuax, nuax], m23, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
        deljy = _compute_delj(dy, MyInt, VyInt, axis=1)
        bc_factory = 0.5
  
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
        by[0,0,0] += (bc_factory/nu2 - My[0,0,0])*2/dy[0]
    if My[-1,-1,-1] >= 0:
        by[-1,-1,-1] += -(-bc_factory/nu2 - My[-1,-1,-1])*2/dy[-1]

    del Vy,VyInt,My,MyInt,deljy

    dz = numpy.diff(zz)
    dfact_z = _compute_dfactor(dz)
    if ploidy3[0]:  
        Vz = _Vfunc(zz, nu3)
        VzInt = _Vfunc((zz[1:]+zz[:-1])/2, nu3)
        Mz = _Mfunc3D(zz[nuax,nuax,:], yy[nuax,:,nuax], xx[:,nuax, nuax],
                      m32, m31, s3[0], s3[1])
        MzInt = _Mfunc3D((zz[nuax,nuax,1:] + zz[nuax,nuax,:-1])/2, yy[nuax,:,nuax],
                          xx[:,nuax, nuax], m32, m31, s3[0], s3[1])
        deljz = _compute_delj(dz, MzInt, VzInt, axis=2)
        bc_factorz = 0.5
    if ploidy3[1]:  
        Vz = _Vfunc_auto(zz, nu3)
        VzInt = _Vfunc_auto((zz[1:]+zz[:-1])/2, nu3)
        Mz = _Mfunc3D_auto(zz[nuax,nuax,:], yy[nuax,:,nuax], xx[:,nuax, nuax],
                      m32, m31, s3[0],s3[1],s3[2],s3[3])
        MzInt = _Mfunc3D_auto((zz[nuax,nuax,1:] + zz[nuax,nuax,:-1])/2, yy[nuax,:,nuax],
                          xx[:,nuax, nuax], m32, m31, s3[0],s3[1],s3[2],s3[3])
        deljz = _compute_delj(dz, MzInt, VzInt, axis=2)
        bc_factorz = 0.25
    if ploidy3[2]:  
        Vz = _Vfunc(zz, nu3)
        VzInt = _Vfunc((zz[1:]+zz[:-1])/2, nu3)
        Mz = _Mfunc3D_allo_a(zz[nuax,nuax,:], yy[nuax,:,nuax], xx[:,nuax, nuax],
                      m32, m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
        MzInt = _Mfunc3D_allo_a((zz[nuax,nuax,1:] + zz[nuax,nuax,:-1])/2, yy[nuax,:,nuax],
                          xx[:,nuax, nuax], m32, m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
        deljz = _compute_delj(dz, MzInt, VzInt, axis=2)
        bc_factorz = 0.5
    if ploidy3[3]:  
        Vz = _Vfunc(zz, nu3)
        VzInt = _Vfunc((zz[1:]+zz[:-1])/2, nu3)
        Mz = _Mfunc3D_allo_b(zz[nuax,nuax,:], yy[nuax,:,nuax], xx[:,nuax, nuax],
                      m32, m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
        MzInt = _Mfunc3D_allo_b((zz[nuax,nuax,1:] + zz[nuax,nuax,:-1])/2, yy[nuax,:,nuax],
                          xx[:,nuax, nuax], m32, m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
        deljz = _compute_delj(dz, MzInt, VzInt, axis=2)
        bc_factorz = 0.5

    az, bz, cz = [numpy.zeros(phi.shape) for ii in range(3)]
    az[:,:, 1:] += dfact_z[ 1:]*(-MzInt*deljz     - Vz[nuax,nuax,:-1]/(2*dz))
    cz[:,:,:-1] += dfact_z[:-1]*( MzInt*(1-deljz) - Vz[nuax,nuax, 1:]/(2*dz))
    bz[:,:,:-1] += dfact_z[:-1]*( MzInt*deljz     + Vz[nuax,nuax,:-1]/(2*dz))
    bz[:,:, 1:] += dfact_z[ 1:]*(-MzInt*(1-deljz) + Vz[nuax,nuax, 1:]/(2*dz))
    if Mz[0,0,0] <= 0:
        bz[0,0,0] += (bc_factorz/nu3 - Mz[0,0,0])*2/dz[0]
    if Mz[-1,-1,-1] >= 0:
        bz[-1,-1,-1] += -(-bc_factorz/nu3 - Mz[-1,-1,-1])*2/dz[-1]

    del Vz,VzInt,Mz,MzInt,deljz

    dt = min(_compute_dt(dx,nu1,[m12,m13],s1,ploidy1),
             _compute_dt(dy,nu2,[m21,m23],s2,ploidy2),
             _compute_dt(dz,nu3,[m31,m32],s3,ploidy3))
    current_t = initial_t
    # TODO: CUDA integration
    if cuda_enabled:
        import dadi.cuda
        phi = dadi.cuda.Integration._three_pops_const_params(phi, xx,
                theta0, frozen1, frozen2, frozen3, 
                ax, bx, cx, ay, by, cy, az, bz, cz,
                current_t, dt, T)
        return phi

    while current_t < T:    
        this_dt = min(dt, T - current_t)
        _inject_mutations_3D(phi, this_dt, xx, yy, zz, theta0,
                             frozen1, frozen2, frozen3)
        if not frozen1:
            phi = int3D.implicit_precalc_3Dx(phi, ax, bx, cx, this_dt)
        if not frozen2:
            phi = int3D.implicit_precalc_3Dy(phi, ay, by, cy, this_dt)
        if not frozen3:
            phi = int3D.implicit_precalc_3Dz(phi, az, bz, cz, this_dt)
        current_t += this_dt
    return phi
