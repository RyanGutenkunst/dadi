import numpy
from numpy import newaxis as nuax

import Numerics, Misc, tridiag
import integration_c as int_c

def inject_mutations_1D(phi, dt, xx, theta0):
    phi[1] += dt/xx[1] * theta0/2 * 2/(xx[2] - xx[0])
    return phi
def inject_mutations_2D(phi, dt, xx, yy, theta0):
    # Population 1
    phi[1,0] += dt/yy[1] * theta0/2 * 4/((yy[2] - yy[0]) * xx[1])
    # Population 2
    phi[0,1] += dt/xx[1] * theta0/2 * 4/((xx[2] - xx[0]) * yy[1])
    return phi
def inject_mutations_3D(phi, dt, xx, yy, zz, theta0):
    # Population 1
    # Normalization based on the multi-dimensional trapezoid rule is 
    # implemented                      ************** here ***************
    phi[1,0,0] += dt/xx[1] * theta0/2 * 8/((xx[2] - xx[0]) * yy[1] * zz[1])
    # Population 2
    phi[0,1,0] += dt/yy[1] * theta0/2 * 8/((yy[2] - yy[0]) * xx[1] * zz[1])
    # Population 3
    phi[0,0,1] += dt/zz[1] * theta0/2 * 8/((zz[2] - zz[0]) * xx[1] * yy[1])
    return phi

def compute_time_steps(T, xx):
    dt = Numerics.timescale_factor*min(numpy.diff(xx))
    time_steps = [dt]*int(T//dt)
    steps_sum = numpy.sum(time_steps)
    if steps_sum < T:
        time_steps.append(T - steps_sum)
    return numpy.array(time_steps)

def one_pop(phi, xx, T, nu=1, gamma=0, theta0=1.0):
    vars_to_check = (nu, gamma, theta0)
    if numpy.all([numpy.isscalar(var) for var in vars_to_check]):
        return one_pop_const_params(phi, xx, T, nu, gamma, theta0)

    nu_f = Misc.ensure_1arg_func(nu)
    gamma_f = Misc.ensure_1arg_func(gamma)
    theta0_f = Misc.ensure_1arg_func(theta0)

    next_t = 0
    time_steps = compute_time_steps(T, xx)
    for this_dt in time_steps:
        next_t += this_dt

        nu, gamma, theta0 = nu_f(next_t), gamma_f(next_t), theta0_f(next_t)
        inject_mutations_1D(phi, this_dt, xx, theta0)
        phi = int_c.implicit_1Dx(phi, xx, nu, gamma, this_dt, 
                                 use_delj_trick=Numerics.use_delj_trick)
    return phi

def two_pops(phi, xx, T, nu1=1, nu2=1, m12=0, m21=0, gamma1=0, gamma2=0,
             theta0=1):
    vars_to_check = [nu1,nu2,m12,m21,gamma1,gamma2,theta0]
    if numpy.all([numpy.isscalar(var) for var in vars_to_check]):
        return two_pops_const_params(phi, xx, T, nu1, nu2, m12, m21, 
                                     gamma1, gamma2, theta0)
    yy = xx

    nu1_f = Misc.ensure_1arg_func(nu1)
    nu2_f = Misc.ensure_1arg_func(nu2)
    m12_f = Misc.ensure_1arg_func(m12)
    m21_f = Misc.ensure_1arg_func(m21)
    gamma1_f = Misc.ensure_1arg_func(gamma1)
    gamma2_f = Misc.ensure_1arg_func(gamma2)
    theta0_f = Misc.ensure_1arg_func(theta0)

    next_t = 0
    time_steps = compute_time_steps(T, xx)
    for ii, this_dt in enumerate(time_steps):
        next_t += this_dt

        nu1,nu2 = nu1_f(next_t), nu2_f(next_t)
        m12,m21 = m12_f(next_t), m21_f(next_t)
        gamma1,gamma2 = gamma1_f(next_t), gamma2_f(next_t)
        theta0 = theta0_f(next_t)

        inject_mutations_2D(phi, this_dt/2, xx, yy, theta0)
        phi = int_c.implicit_2Dx(phi, xx, yy, nu1, m12, gamma1,
                                 this_dt, Numerics.use_delj_trick)

        inject_mutations_2D(phi, this_dt/2, xx, yy, theta0)
        phi = int_c.implicit_2Dy(phi, xx, yy, nu2, m21, gamma2, 
                                 this_dt, Numerics.use_delj_trick)
    return phi

def three_pops(phi, xx, T, nu1=1, nu2=1, nu3=1,
               m12=0, m13=0, m21=0, m23=0, m31=0, m32=0,
               gamma1=0, gamma2=0, gamma3=0, theta0=1):
    vars_to_check = [nu1,nu2,nu3,m12,m13,m21,m23,m31,m32,gamma1,gamma2,
                     gamma3,theta0]
    if numpy.all([numpy.isscalar(var) for var in vars_to_check]):
        return three_pops_const_params(phi, xx, T, nu1, nu2, nu3, 
                                       m12, m13, m21, m23, m31, m32, 
                                       gamma1, gamma2, gamma3, theta0)
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
    theta0_f = Misc.ensure_1arg_func(theta0)

    next_t = 0
    time_steps = compute_time_steps(T, xx)
    for this_dt in time_steps:
        next_t += this_dt

        nu1,nu2,nu3 = nu1_f(next_t), nu2_f(next_t), nu3_f(next_t)
        m12,m13 = m12_f(next_t), m13_f(next_t)
        m21,m23 = m21_f(next_t), m23_f(next_t)
        m31,m32 = m31_f(next_t), m32_f(next_t)
        gamma1,gamma2 = gamma1_f(next_t), gamma2_f(next_t)
        gamma3 = gamma3_f(next_t)
        theta0 = theta0_f(next_t)

        inject_mutations_3D(phi, this_dt/3, xx, yy, zz, theta0)
        phi = int_c.implicit_3Dx(phi, xx, yy, zz, nu1, m12, m13, 
                                 gamma1, this_dt, Numerics.use_delj_trick)
        inject_mutations_3D(phi, this_dt/3, xx, yy, zz, theta0)
        phi = int_c.implicit_3Dy(phi, xx, yy, zz, nu2, m21, m23, 
                                 gamma2, this_dt, Numerics.use_delj_trick)
        inject_mutations_3D(phi, this_dt/3, xx, yy, zz, theta0)
        phi = int_c.implicit_3Dz(phi, xx, yy, zz, nu3, m31, m32, 
                                 gamma3, this_dt, Numerics.use_delj_trick)
    return phi                                                      
#
# Here are the python versions of the population genetic functions.
#
def Vfunc(x, nu):
    return 1./nu * x*(1-x)
def Mfunc1D(x, gamma):
    return gamma * x*(1-x)
def Mfunc2D(x,y,mxy,gammax):
    return mxy * (y-x) + gammax * x*(1-x)
def Mfunc3D(x,y,z,mxy,mxz,gammax):
    return mxy * (y-x) + mxz * (z-x) + gammax * x*(1-x)

def compute_dfactor(dx):
    # Controls how we take the derivative of the flux. The values here depend
    #  on the fact that we're defining our probability integral using the
    #  trapezoid rule.
    dfactor = numpy.zeros(len(dx)+1)
    dfactor[1:-1] = 2/(dx[:-1] + dx[1:])
    dfactor[0] = 2/dx[0]
    dfactor[-1] = 2/dx[-1]
    return dfactor

def compute_delj(dx, MInt, VInt, axis=0):
    # Chang and Cooper's fancy delta j trick...
    if Numerics.use_delj_trick:
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

def one_pop_const_params(phi, xx, T, nu=1, gamma=0, theta0=1):
    M = Mfunc1D(xx, gamma)
    MInt = Mfunc1D((xx[:-1] + xx[1:])/2, gamma)
    V = Vfunc(xx, nu)
    VInt = Vfunc((xx[:-1] + xx[1:])/2, nu)

    dx = numpy.diff(xx)
    dfactor = compute_dfactor(dx)
    delj = compute_delj(dx, MInt, VInt)

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

    time_steps = compute_time_steps(T, xx)
    for this_dt in time_steps:
        inject_mutations_1D(phi, this_dt, xx, theta0)
        r = phi/this_dt
        phi = tridiag.tridiag(a, b+1/this_dt, c, r)

    return phi

def two_pops_const_params(phi, xx, T, nu1=1, nu2=1, m12=0, m21=0,
                          gamma1=0, gamma2=0, theta0=1):
    yy = xx

    # The use of nuax (= numpy.newaxis) here is for memory conservation. We
    # could just create big X and Y arrays which only varied along one axis,
    # but that would be wasteful.
    Vx = Vfunc(xx, nu1)
    VxInt = Vfunc((xx[:-1]+xx[1:])/2, nu1)
    Mx = Mfunc2D(xx[:,nuax], yy[nuax,:], m12, gamma1)
    MxInt = Mfunc2D((xx[:-1,nuax]+xx[1:,nuax])/2, yy[nuax,:], m12, gamma1)

    Vy = Vfunc(yy, nu2)
    VyInt = Vfunc((yy[1:]+yy[:-1])/2, nu2)
    My = Mfunc2D(yy[nuax,:], xx[:,nuax], m21, gamma2)
    MyInt = Mfunc2D((yy[nuax,1:] + yy[nuax,:-1])/2, xx[:,nuax], m21, gamma2)

    dx = numpy.diff(xx)
    dfact_x = compute_dfactor(dx)
    deljx = compute_delj(dx, MxInt, VxInt)

    dy = numpy.diff(yy)
    dfact_y = compute_dfactor(dy)
    deljy = compute_delj(dy, MyInt, VyInt, axis=1)

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

    time_steps = compute_time_steps(T, xx)
    for this_dt in time_steps:
        inject_mutations_2D(phi, this_dt/2, xx, yy, theta0)
        phi = int_c.implicit_precalc_2Dx(phi, ax, bx, cx, this_dt)
        inject_mutations_2D(phi, this_dt/2, xx, yy, theta0)
        phi = int_c.implicit_precalc_2Dy(phi, ay, by, cy, this_dt)

    return phi

def three_pops_const_params(phi, xx, T, nu1=1, nu2=1, nu3=1, 
                            m12=0, m13=0, m21=0, m23=0, m31=0, m32=0, 
                            gamma1=0, gamma2=0, gamma3=0, theta0=1):
    zz = yy = xx

    Vx = Vfunc(xx, nu1)
    VxInt = Vfunc((xx[:-1]+xx[1:])/2, nu1)
    Mx = Mfunc3D(xx[:,nuax,nuax], yy[nuax,:,nuax], zz[nuax,nuax,:], 
                 m12, m13, gamma1)
    MxInt = Mfunc3D((xx[:-1,nuax,nuax]+xx[1:,nuax,nuax])/2, yy[nuax,:,nuax], 
                    zz[nuax,nuax,:], m12, m13, gamma1)

    dx = numpy.diff(xx)
    dfact_x = compute_dfactor(dx)
    deljx = compute_delj(dx, MxInt, VxInt)

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

    Vy = Vfunc(yy, nu2)
    VyInt = Vfunc((yy[1:]+yy[:-1])/2, nu2)
    My = Mfunc3D(yy[nuax,:,nuax], xx[:,nuax, nuax], zz[nuax,nuax,:],
                 m21, m23, gamma2)
    MyInt = Mfunc3D((yy[nuax,1:,nuax] + yy[nuax,:-1,nuax])/2, xx[:,nuax, nuax], 
                    zz[nuax,nuax,:], m21, m23, gamma2)

    dy = numpy.diff(yy)
    dfact_y = compute_dfactor(dy)
    deljy = compute_delj(dy, MyInt, VyInt, axis=1)

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

    Vz = Vfunc(zz, nu3)
    VzInt = Vfunc((zz[1:]+zz[:-1])/2, nu3)
    Mz = Mfunc3D(zz[nuax,nuax,:], xx[:,nuax, nuax], yy[nuax,:,nuax],
                 m31, m32, gamma3)
    MzInt = Mfunc3D((zz[nuax,nuax,1:] + zz[nuax,nuax,:-1])/2, xx[:,nuax, nuax], 
                    yy[nuax,:,nuax], m31, m32, gamma3)

    dz = numpy.diff(zz)
    dfact_z = compute_dfactor(dz)
    deljz = compute_delj(dz, MzInt, VzInt, axis=2)

    az, bz, cz = [numpy.zeros(phi.shape) for ii in range(3)]
    az[:,:, 1:] += dfact_z[ 1:]*(-MzInt*deljz     - Vz[nuax,nuax,:-1]/(2*dz))
    cz[:,:,:-1] += dfact_z[:-1]*( MzInt*(1-deljz) - Vz[nuax,nuax, 1:]/(2*dz))
    bz[:,:,:-1] += dfact_z[:-1]*( MzInt*deljz     + Vz[nuax,nuax,:-1]/(2*dz))
    bz[:,:, 1:] += dfact_z[ 1:]*(-MzInt*(1-deljz) + Vz[nuax,nuax, 1:]/(2*dz))

    if Mz[0,0,0] <= 0:
        bz[0,0,0] += (0.5/nu3 - Mz[0,0,0])*2/dz[0]
    if Mz[-1,-1,-1] >= 0:
        bz[-1,-1,-1] += -(-0.5/nu3 - Mz[-1,-1,-1])*2/dz[-1]

    time_steps = compute_time_steps(T, xx)
    for this_dt in time_steps:
        inject_mutations_3D(phi, this_dt/3, xx, yy, zz, theta0)
        phi = int_c.implicit_precalc_3Dx(phi, ax, bx, cx, this_dt)

        inject_mutations_3D(phi, this_dt/3, xx, yy, zz, theta0)
        phi = int_c.implicit_precalc_3Dy(phi, ay, by, cy, this_dt)

        inject_mutations_3D(phi, this_dt/3, xx, yy, zz, theta0)
        phi = int_c.implicit_precalc_3Dz(phi, az, bz, cz, this_dt)
    return phi