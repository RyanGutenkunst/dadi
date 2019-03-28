import numpy as np
import dadi
from scipy.sparse import identity
from . import numerics

def advance(phi, x, T, yA, yB, nu = 1., gammaA = 0.0, gammaB = 0.0, hA = 0.5, hB = 0.5, rho = 0.0, thetaA = 1.0, thetaB = 1.0, dt = 0.001):
    """
    Integrate phi, yA, and yB forward in time, using dadi for the 
    biallelic density functions and numerics methods for phi
    """
    dx = numerics.grid_dx(x)
    dx3 = numerics.grid_dx3(x,dx)
    U01 = numerics.domain(x)

    P1 = np.outer([0,1,0],np.ones(len(x))) + dt * numerics.transition1(x, dx, U01, gammaA, gammaB, rho, nu, hA=hA, hB=hB)
    P2 = np.outer([0,1,0],np.ones(len(x))) + dt * numerics.transition2(x, dx, U01, gammaA, gammaB, rho, nu, hA=hA, hB=hB)
    P3 = np.outer([0,1,0],np.ones(len(x))) + dt * numerics.transition3(x, dx, U01, gammaA, gammaB, rho, nu, hA=hA, hB=hB)

    C12 = numerics.transition12(x,dx,U01)
    for kk in range(len(x)):
        C12[kk] = identity(len(x)**2) + dt/nu*C12[kk]

    C13 = numerics.transition13(x,dx,U01)
    for kk in range(len(x)):
        C13[kk] = identity(len(x)**2) + dt/nu*C13[kk]

    C23 = numerics.transition23(x,dx,U01)
    for kk in range(len(x)):
        C23[kk] = identity(len(x)**2) + dt/nu*C23[kk]

    Psurf = numerics.move_density_to_surface(x, dx, dt, gammaA, gammaB, nu, hA=hA, hB=hB)

    # surface transition matrices
    
    #### 9/11 still need to incorporate dominance into surface integration
    
    
    U01surf = numerics.domain_surf(x)
    P1surf =  np.outer([0,1,0],np.ones(len(x))) + dt * numerics.transition1_surf(x, dx, U01surf, gammaA, gammaB, rho, nu, hA=hA, hB=hB)
    P2surf =  np.outer([0,1,0],np.ones(len(x))) + dt * numerics.transition2_surf(x, dx, U01surf, gammaA, gammaB, rho, nu, hA=hA, hB=hB)
    Csurf = identity(len(x)**2) + dt/nu*numerics.transition12_surf(x, dx, U01surf)
    Pline = np.eye(len(x))
    P = numerics.move_surf_to_line(x, dx, dt, gammaA, gammaB, nu)
    #Pline = numerics.transition1D(x, dx, dt, gamma, nu)
    
    if np.all(phi == 0) and T >= 5: # solving to equilibrium - integrate at first without covariance term so that the surface is smooth first
        yA,yB,phi = advance_without_cov(phi,x,dx,dt,yA,yB,thetaA,thetaB,U01,P1,P2,P3,Psurf,rho,P1surf,P2surf,Pline,P,U01surf,1.)
        
    for ii in range(int(T/dt)):
        # integrate the biallelic density functions forward in time using dadi
        yA = dadi.Integration.one_pop(yA, x, dt, nu = nu, gamma = gammaA, theta0 = thetaA)
        yB = dadi.Integration.one_pop(yB, x, dt, nu = nu, gamma = gammaB, theta0 = thetaB)

        # inject new mutations using biallelic density functions
        phi = numerics.injectA(x, dx, dt, yB, phi, thetaA) # A is injected onto B background, so need yB
        phi = numerics.injectB(x, dx, dt, yA, phi, thetaB) # B is injected onto A background, so need yA

        # advance bulk of phi forward by dt using numerics methods advance_adi and advance_cov
        phi = numerics.advance_adi(phi, U01, P1, P2, P3, x, ii)
        phi = numerics.advance_cov(phi, C12, C13, C23, x, ii)

        # methods for interaction with and integration of non-axis surface
        phi = numerics.surface_interaction(phi,x,Psurf)
        phi = numerics.advance_surface(phi,x,P1surf,P2surf,Csurf,Pline,P,U01surf)
        phi = numerics.surface_recombination(phi,x,rho/2.,dt) ## changed to rho/2 5/29 - note that this is effectively only "half" of the recombination events. the other half are along the surface.
        
    return yA, yB, phi

def advance_without_cov(phi,x,dx,dt,yA,yB,thetaA,thetaB,U01,P1,P2,P3,Psurf,rho,P1surf,P2surf,Pline,P,U01surf,T):
    Csurf = identity(len(x)**2)
    for ii in range(int(T/dt)):
        phi = numerics.injectA(x, dx, dt, yB, phi, thetaA)
        phi = numerics.injectB(x, dx, dt, yA, phi, thetaB)
        phi = numerics.advance_adi(phi, U01, P1, P2, P3, x,ii)
        phi = numerics.surface_interaction(phi,x,Psurf) 
        phi = numerics.advance_surface(phi,x,P1surf,P2surf,Csurf,Pline,P,U01surf)
        phi = numerics.surface_recombination(phi,x,rho/2.,dt)
    return yA,yB,phi



###


def advance_injection_test(phi, x, T, yA, yB, nu = 1., gammaA = 0.0, gammaB = 0.0, rho = 0.0, thetaA = 1.0, thetaB = 1.0, dt = 0.001):
    """
    Integrate phi, yA, and yB forward in time, using dadi for the 
    biallelic density functions and numerics methods for phi
    """
    dx = numerics.grid_dx(x)
    dx3 = numerics.grid_dx3(x,dx)
    U01 = numerics.domain(x)
        
    for ii in range(int(T/dt)):
        # integrate the biallelic density functions forward in time using dadi
        yA = dadi.Integration.one_pop(yA, x, dt, nu = nu, gamma = gammaA, theta0 = thetaA)
        yB = dadi.Integration.one_pop(yB, x, dt, nu = nu, gamma = gammaB, theta0 = thetaB)

        # inject new mutations using biallelic density functions
        phi = numerics.injectA(x, dx, dt, yA, phi, thetaA)
        phi = numerics.injectB(x, dx, dt, yB, phi, thetaB)
        
    return yA, yB, phi

