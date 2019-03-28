"""
Integration of phi for triallelic diffusion
These methods include ones for the injection of density for new triallelic sites and integration forward in time
"""

import numpy as np
from numpy import newaxis as nuax
import dadi
from . import numerics
from scipy.sparse import identity
import scipy.io
import math

def inject_mutations_1(phi, dt, x, dx, y2, theta1):
    """
    new mutations injected along phi[1,:] against a background given by y2
    phi - numerical density function
    dt - given time step 
    x, dx - one dimensional grid and grid spacing
    y2 - the biallelic density function
    theta1 - population scaled mutation rate for mutation 1
    """
    phi[1,1:-1] += y2[1:-1] / dx[1] * 1./x[1] * dt * theta1/2
    return phi

def inject_mutations_2(phi, dt, x, dx, y1, theta2):
    """
    new mutations injected along phi[:,1] against a background given by y1
    phi - numerical density function
    dt - given time step 
    x, dx - one dimensional grid and grid spacing
    y1 - the biallelic density function
    theta2 - population scaled mutation rate for mutation 2
    """
    phi[1:-1,1] += y1[1:-1] / dx[1] * 1./x[1] * dt * theta2/2
    return phi

def inject_simultaneous_muts(phi, dt, x, dx, theta):
    """
    simultaneous mutation model - see Hodgkinson and Eyre-Walker 2010, injected at (Delta,Delta)
    """
    phi[1,1] += 1. / x[1] / x[1] / dx[1] / dx[1] * dt * theta
    return phi

def equilibrium_neutral_exact(x):
    """
    With thetas = 1
    nu = 1
    sig1 = sig2 = 0
    """
    phi = np.zeros((len(x),len(x)))
    for ii in range(len(phi))[1:]:
        phi[ii,1:-ii-1] = 1./x[ii]/x[1:-ii-1]
    return phi

def advance(phi, x, T, y1, y2, nu=1., sig1=0., sig2=0., theta1=1., theta2=1., dt=0.001):
    """
    Integrate phi, y1, and y2 forward in time
    phi - density function for triallelic sites
    y1,y2 - density of biallelic background sites, integrated forward alongside phi
    T - amount of time to integrate, scaled by 2N generations
    nu - relative size of population to ancestral size
    sig1,sig2 - selection coefficients for two derived alleles
    theta1,theta2 - population scaled mutation rates
    dt - time step for integration
    lam - proportion of mutations that occur from simulateous mutation model (Hodgkinson/Eyre-Walker 2010)
    """
    dx = numerics.grid_dx(x)
    U01 = numerics.domain(x)
    
    C_base = numerics.transition12(x,dx,U01)
    V1_base,M1_base = numerics.transition1(x,dx,U01,sig1,sig2)
    V2_base,M2_base = numerics.transition2(x,dx,U01,sig1,sig2)
    V1D1_base,M1D1_base = numerics.transition1D(x,dx,sig1)
    V1D2_base,M1D2_base = numerics.transition1D(x,dx,sig2)
    sig_line = sig1-sig2
    Vline_base,Mline_base = numerics.transition1D(x,dx,sig_line)

    if np.isscalar(nu):
        C = identity(len(x)**2) + dt/nu*C_base
        P1 = np.outer(np.array([0,1,0]),np.ones(len(x))) + dt*(V1_base/nu+M1_base)
        P2 = np.outer(np.array([0,1,0]),np.ones(len(x))) + dt*(V2_base/nu+M2_base)
        P1D1 = np.eye(len(x)) + dt*(V1D1_base/nu+M1D1_base)
        P1D2 = np.eye(len(x)) + dt*(V1D2_base/nu+M1D2_base)
        Pline = np.eye(len(x)) + dt*(Vline_base/nu+Mline_base)
        P = numerics.remove_diag_density_weights_nonneutral(x,dt,nu,sig1,sig2)

        for ii in range(int(T/dt)):
            y1[1] += dt/dx[1]/x[1]/2 * theta1
            y1 = numerics.advance1D(y1,P1D1)
            y2[1] += dt/dx[1]/x[1]/2 * theta2
            y2 = numerics.advance1D(y2,P1D2)
            phi = inject_mutations_1(phi, dt, x, dx, y2, theta1)
            phi = inject_mutations_2(phi, dt, x, dx, y1, theta2)
            phi = numerics.advance_adi(phi,U01,P1,P2,x,ii)
            phi = numerics.advance_cov(phi,C,x,dx)
            #phi *= 1-P
            # move density to diagonal boundary and integrate it
            phi = numerics.move_density_to_bdry(x,phi,P)
            phi = numerics.advance_line(x,phi,Pline)
        
        T_elapsed = int(T/dt)*dt
        if T - T_elapsed > 1e-8:
            # adjust dt and integrate last time step
            dt = T-T_elapsed
            C = identity(len(x)**2) + dt/nu*C_base
            P1 = np.outer(np.array([0,1,0]),np.ones(len(x))) + dt*(V1_base/nu+M1_base)
            P2 = np.outer(np.array([0,1,0]),np.ones(len(x))) + dt*(V2_base/nu+M2_base)
            P1D1 = np.eye(len(x)) + dt*(V1D1_base/nu+M1D1_base)
            P1D2 = np.eye(len(x)) + dt*(V1D2_base/nu+M1D2_base)
            Pline = np.eye(len(x)) + dt*(Vline_base/nu+Mline_base)
            P = numerics.remove_diag_density_weights_nonneutral(x,dt,nu,sig1,sig2)
            
            y1[1] += dt/dx[1]/x[1]/2 * theta1
            y1 = numerics.advance1D(y1,P1D1)
            y2[1] += dt/dx[1]/x[1]/2 * theta2
            y2 = numerics.advance1D(y2,P1D2)
            phi = inject_mutations_1(phi, dt, x, dx, y2, theta1)
            phi = inject_mutations_2(phi, dt, x, dx, y1, theta2)
            phi = numerics.advance_adi(phi,U01,P1,P2,x,0)
            phi = numerics.advance_cov(phi,C,x,dx)
            #phi *= 1-P
            # move density to diagonal boundary and integrate it
            phi = numerics.move_density_to_bdry(x,phi,P)
            phi = numerics.advance_line(x,phi,Pline)
    else:
        Ts = np.concatenate(( np.linspace(0,np.floor(T/dt)*dt,np.floor(T/dt)+1), np.array([T]) ))
        
        for ii in range(int(T/dt)):
            nu_current = nu(Ts[ii])
            C = identity(len(x)**2) + dt/nu_current*C_base
            P1 = np.outer(np.array([0,1,0]),np.ones(len(x))) + dt*(V1_base/nu_current+M1_base)
            P2 = np.outer(np.array([0,1,0]),np.ones(len(x))) + dt*(V2_base/nu_current+M2_base)
            P1D1 = np.eye(len(x)) + dt*(V1D1_base/nu_current+M1D1_base)
            P1D2 = np.eye(len(x)) + dt*(V1D2_base/nu_current+M1D2_base)
            Pline = np.eye(len(x)) + dt*(Vline_base/nu_current+Mline_base)
            P = numerics.remove_diag_density_weights_nonneutral(x,dt,nu_current,sig1,sig2)                
            
            y1[1] += dt/dx[1]/x[1]/2 * theta1
            y1 = numerics.advance1D(y1,P1D1)
            y2[1] += dt/dx[1]/x[1]/2 * theta2
            y2 = numerics.advance1D(y2,P1D2)
            phi = inject_mutations_1(phi, dt, x, dx, y2, theta1)
            phi = inject_mutations_2(phi, dt, x, dx, y1, theta2)
            phi = numerics.advance_adi(phi,U01,P1,P2,x,ii)
            phi = numerics.advance_cov(phi,C,x,dx)
            #phi *= 1-P
            # move density to diagonal boundary and integrate it
            phi = numerics.move_density_to_bdry(x,phi,P)
            phi = numerics.advance_line(x,phi,Pline)
        
        T_elapsed = int(T/dt)*dt
        if T - T_elapsed > 1e-8:
            # adjust dt and integrate last time step
            dt = T-T_elapsed
            nu_current = nu(Ts[-1])
            C = identity(len(x)**2) + dt/nu_current*C_base
            P1 = np.outer(np.array([0,1,0]),np.ones(len(x))) + dt*(V1_base/nu_current+M1_base)
            P2 = np.outer(np.array([0,1,0]),np.ones(len(x))) + dt*(V2_base/nu_current+M2_base)
            P1D1 = np.eye(len(x)) + dt*(V1D1_base/nu_current+M1D1_base)
            P1D2 = np.eye(len(x)) + dt*(V1D2_base/nu_current+M1D2_base)
            Pline = np.eye(len(x)) + dt*(Vline_base/nu_current+Mline_base)
            P = numerics.remove_diag_density_weights_nonneutral(x,dt,nu_current,sig1,sig2)                
            
            y1[1] += dt/dx[1]/x[1]/2 * theta1
            y1 = numerics.advance1D(y1,P1D1)
            y2[1] += dt/dx[1]/x[1]/2 * theta2
            y2 = numerics.advance1D(y2,P1D2)
            phi = inject_mutations_1(phi, dt, x, dx, y2, theta1)
            phi = inject_mutations_2(phi, dt, x, dx, y1, theta2)
            phi = numerics.advance_adi(phi,U01,P1,P2,x,0)
            phi = numerics.advance_cov(phi,C,x,dx)
            #phi *= 1-P
            # move density to diagonal boundary and integrate it
            phi = numerics.move_density_to_bdry(x,phi,P)
            phi = numerics.advance_line(x,phi,Pline)

    return phi,y1,y2


def advance_old(phi, x, T, y1, y2, nu=1., sig1=0., sig2=0., theta1=1., theta2=1., dt=0.001):
    """
    Integrate phi, y1, and y2 forward in time
    phi - density function for triallelic sites
    y1,y2 - density of biallelic background sites, integrated forward alongside phi
    T - amount of time to integrate, scaled by 2N generations
    nu - relative size of population to ancestral size
    sig1,sig2 - selection coefficients for two derived alleles
    theta1,theta2 - population scaled mutation rates
    dt - time step for integration
    lam - proportion of mutations that occur from simulateous mutation model (Hodgkinson/Eyre-Walker 2010)
    """
    dx = numerics.grid_dx(x)
    U01 = numerics.domain(x)
    C = identity(len(x)**2) + dt/nu*numerics.transition12(x,dx,U01)
    P1 = np.outer(np.array([0,1,0]),np.ones(len(x))) + dt*numerics.transition1(x,dx,U01,sig1,sig2,nu) 
    P2 = np.outer(np.array([0,1,0]),np.ones(len(x))) + dt*numerics.transition2(x,dx,U01,sig1,sig2,nu)
    P1D1 = numerics.transition1D(x,dx,dt,sig1,nu)
    P1D2 = numerics.transition1D(x,dx,dt,sig2,nu)
    
    sig_line = sig1-sig2
    Pline = numerics.transition1D(x,dx,dt,sig_line,nu)
    P = numerics.remove_diag_density_weights_nonneutral(x,dt,nu,sig1,sig2)

    for ii in range(int(T/dt)):
        y1[1] += dt/dx[1]/x[1]/2 * theta1
        y1 = numerics.advance1D(y1,P1D1)
        y2[1] += dt/dx[1]/x[1]/2 * theta2
        y2 = numerics.advance1D(y2,P1D2)
        phi = inject_mutations_1(phi, dt, x, dx, y2, theta1)
        phi = inject_mutations_2(phi, dt, x, dx, y1, theta2)
        phi = numerics.advance_adi(phi,U01,P1,P2,x,ii)
        phi = numerics.advance_cov(phi,C,x,dx)
        #phi *= 1-P
        # move density to diagonal boundary and integrate it
        phi = numerics.move_density_to_bdry(x,phi,P)
        phi = numerics.advance_line(x,phi,Pline)
    
    T_elapsed = int(T/dt)*dt
    if T - T_elapsed > 1e-8:
        # adjust dt and integrate last time step
        dt = T-T_elapsed
        C = identity(len(x)**2) + dt/nu*numerics.transition12(x,dx,U01) # covariance term
        P1 = np.outer(np.array([0,1,0]),np.ones(len(x))) + dt*numerics.transition1(x,dx,U01,sig1,sig2,nu) 
        P2 = np.outer(np.array([0,1,0]),np.ones(len(x))) + dt*numerics.transition2(x,dx,U01,sig1,sig2,nu)
        P1D1 = numerics.transition1D(x,dx,dt,sig1,nu)
        P1D2 = numerics.transition1D(x,dx,dt,sig2,nu)
        
        sig_line = sig1-sig2
        Pline = numerics.transition1D(x,dx,dt,sig_line,nu)
        P = numerics.remove_diag_density_weights_nonneutral(x,dt,nu,sig1,sig2)
        
        y1[1] += dt/dx[1]/x[1]/2 * theta1
        y1 = numerics.advance1D(y1,P1D1)
        y2[1] += dt/dx[1]/x[1]/2 * theta2
        y2 = numerics.advance1D(y2,P1D2)
        phi = inject_mutations_1(phi, dt, x, dx, y2, theta1)
        phi = inject_mutations_2(phi, dt, x, dx, y1, theta2)
        phi = numerics.advance_adi(phi,U01,P1,P2,x,0)
        phi = numerics.advance_cov(phi,C,x,dx)
        #phi *= 1-P
        # move density to diagonal boundary and integrate it
        phi = numerics.move_density_to_bdry(x,phi,P)
        phi = numerics.advance_line(x,phi,Pline)
    
    return phi,y1,y2

def alt_mut_mech_sample_spectrum(ns):
    """
    alternate mutation mechanism, mutations inserted at [1,1]
    turns out that changing population size does not effect the distribution of mutations entering the population this way
    we implement Jenkins et al (2014) exact solution
    this is for neutral spectrum only, for selected spectrum, integrate as above with lam = 1
    ns - number of sampled individuals from the population
    """
    fs = np.zeros((ns+1,ns+1))
    for ii in range(ns)[1:]:
        for jj in range(ns)[1:]:
            if ii + jj < ns:
                na = ns - ii - jj
                fs[ii,jj] = 2*ns/(ns-2) * 1./((ns-na-1)*(ns-na)*(ns-na+1))
    fs = dadi.Spectrum(fs)
    fs[:,0].mask = True
    fs[0,:].mask = True
    for ii in range(len(fs)):
        fs.mask[ii,ns-ii:] = True
    return fs
