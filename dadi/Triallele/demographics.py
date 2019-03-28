"""
Equilibrium, two epoch, and three epoch (bottleneck) model with selection (sig1, sig2) on the two derived alleles
"""

from . import numerics, integration
import numpy as np
import dadi
from numpy import newaxis as nuax

from dadi.Triallele.TriSpectrum_mod import TriSpectrum

def equilibrium(params, ns, pts, sig1 = 0.0, sig2 = 0.0, theta1 = 1.0, theta2 = 1.0, misid = 0.0, dt = 0.005, folded = False):
    """
    Integrate the density function to equilibrium
    params = unused
    sig1, sig2 - population scaled selection coefficients for the two derived alleles
    theta1, theta2 - population scaled mutation rates
    misid - ancestral misidentification parameter
    dt - time step to use for integration
    folded = True - fold the frequency spectrum (if we assume we don't know the order that derived alleles appeared)
    """
    x = np.linspace(0,1,pts+1)
    sig1,sig2 = np.float(sig1),np.float(sig2)

    y1 = dadi.PhiManip.phi_1D(x,gamma=sig1)
    y2 = dadi.PhiManip.phi_1D(x,gamma=sig2)
    phi = np.zeros((len(x),len(x)))
    if sig1 == sig2 == 0.0 and theta1 == theta2 == 1:
        phi = integration.equilibrium_neutral_exact(x)
    else:
        phi = integration.equilibrium_neutral_exact(x)
        phi,y1,y2 = integration.advance(phi, x, 2, y1, y2, nu=1., sig1=sig1, sig2=sig2, theta1=theta1, theta2=theta2, dt=dt)

    dx = numerics.grid_dx(x)

    try:
        ns = int(ns)
    except TypeError:
        ns = ns[0]

    fs = numerics.sample(phi, ns, x)
    fs.extrap_t = dt

    if folded == True:
        fs = fs.fold_major()

    if misid > 0.0:
        fs = numerics.misidentification(fs,misid)
    
    return fs
    
def two_epoch(params, ns, pts, sig1 = 0.0, sig2 = 0.0, theta1 = 1.0, theta2 = 1.0, misid = 0.0, dt = 0.005, folded = False):
    """
    Two epoch demography - a single population size change at some point in the past
    params = [nu,T,sig1,sig2,theta1,theta2,misid,dt]
    nu - relative poplulation size change to ancestral population size
    T - time in past that size change occured (scaled by 2N generations)
    sig1, sig2 - population scaled selection coefficients for the two derived alleles
    theta1, theta2 - population scaled mutation rates
    misid - ancestral misidentification parameter
    dt - time step to use for integration
    """
    nu,T = params

    x = np.linspace(0,1,pts+1)
    sig1,sig2 = np.float(sig1),np.float(sig2)

    y1 = dadi.PhiManip.phi_1D(x,gamma=sig1)
    y2 = dadi.PhiManip.phi_1D(x,gamma=sig2)
    phi = np.zeros((len(x),len(x)))

    # integrate to equilibrium first
    if sig1 == sig2 == 0.0 and theta1 == theta2 == 1:
        phi = integration.equilibrium_neutral_exact(x)
    else:
        phi = integration.equilibrium_neutral_exact(x)
        phi,y1,y2 = integration.advance(phi, x, 2, y1, y2, nu=1., sig1=sig1, sig2=sig2, theta1=theta1, theta2=theta2, dt=dt)

    phi,y1,y2 = integration.advance(phi, x, T, y1, y2, nu=nu, sig1=sig1, sig2=sig2, theta1=theta1, theta2=theta2, dt=dt)

    dx = numerics.grid_dx(x)

    try:
        ns = int(ns)
    except TypeError:
        ns = ns[0]

    fs = numerics.sample(phi, ns, x)
    fs.extrap_t = dt

    if folded == True:
        fs = fs.fold_major()
    
    if misid > 0.0:
        fs = numerics.misidentification(fs,misid)
    
    return fs


def three_epoch(params, ns, pts, sig1 = 0.0, sig2 = 0.0, theta1 = 1.0, theta2 = 1.0, misid = 0.0, dt = 0.005, folded = False):
    """
    Three epoch demography - two instantaneous population size changes in the past
    params = [nu1,nu2,T1,T2]
    nu1,nu2 - relative poplulation size changes to ancestral population size (nu1 occurs before nu2, historically)
    T1,T2 - time for which population had relative sizes nu1, nu2 (scaled by 2N generations)
    sig1, sig2 - population scaled selection coefficients for the two derived alleles
    theta1, theta2 - population scaled mutation rates
    misid - ancestral misidentification parameter
    dt - time step to use for integration
    """
    nu1,nu2,T1,T2 = params
    x = np.linspace(0,1,pts+1)
    sig1,sig2 = np.float(sig1),np.float(sig2)
    
    y1 = dadi.PhiManip.phi_1D(x,gamma=sig1)
    y2 = dadi.PhiManip.phi_1D(x,gamma=sig2)
    phi = np.zeros((len(x),len(x)))
    
    # integrate to equilibrium first
    if sig1 == sig2 == 0.0 and theta1 == theta2 == 1:
        phi = integration.equilibrium_neutral_exact(x)
    else:
        phi = integration.equilibrium_neutral_exact(x)
        phi,y1,y2 = integration.advance(phi, x, 2, y1, y2, nu=1., sig1=sig1, sig2=sig2, theta1=theta1, theta2=theta2, dt=dt)
    
    phi,y1,y2 = integration.advance(phi, x, T1, y1, y2, nu1, sig1, sig2, theta1, theta2, dt=dt)
    phi,y1,y2 = integration.advance(phi, x, T2, y1, y2, nu2, sig1, sig2, theta1, theta2, dt=dt)

    dx = numerics.grid_dx(x)
    
    try:
        ns = int(ns)
    except TypeError:
        ns = ns[0]
    
    fs = numerics.sample(phi, ns, x)
    fs.extrap_t = dt

    if folded == True:
        fs = fs.fold_major()
    
    if misid > 0.0:
        fs = numerics.misidentification(fs,misid)
    
    return fs

def bottlegrowth(params, ns, pts, sig1 = 0.0, sig2 = 0.0, theta1 = 1.0, theta2 = 1.0, misid = 0.0, dt = 0.005, folded = False):
    """
    Three epoch demography - two instantaneous population size changes in the past
    params = [nu1,nu2,T1,T2]
    nu1,nu2 - relative poplulation size changes to ancestral population size (nu1 occurs before nu2, historically)
    T1,T2 - time for which population had relative sizes nu1, nu2 (scaled by 2N generations)
    sig1, sig2 - population scaled selection coefficients for the two derived alleles
    theta1, theta2 - population scaled mutation rates
    misid - ancestral misidentification parameter
    dt - time step to use for integration
    """
    nuB,nuF,T = params
    if nuB == nuF:
        nu = nuB
    else:
        nu = lambda t: nuB*np.exp(np.log(nuF/nuB) * t/T)
    
    x = np.linspace(0,1,pts+1)
    sig1,sig2 = np.float(sig1),np.float(sig2)
    
    y1 = dadi.PhiManip.phi_1D(x,gamma=sig1)
    y2 = dadi.PhiManip.phi_1D(x,gamma=sig2)
    phi = np.zeros((len(x),len(x)))
    
    # integrate to equilibrium first
    if sig1 == sig2 == 0.0 and theta1 == theta2 == 1:
        phi = integration.equilibrium_neutral_exact(x)
    else:
        phi = integration.equilibrium_neutral_exact(x)
        phi,y1,y2 = integration.advance(phi, x, 2, y1, y2, nu=1., sig1=sig1, sig2=sig2, theta1=theta1, theta2=theta2, dt=dt)
    
    phi,y1,y2 = integration.advance(phi, x, T, y1, y2, nu, sig1, sig2, theta1, theta2, dt=dt)

    dx = numerics.grid_dx(x)
    
    try:
        ns = int(ns)
    except TypeError:
        ns = ns[0]
    
    fs = numerics.sample(phi, ns, x)
    fs.extrap_t = dt

    if folded == True:
        fs = fs.fold_major()
    
    if misid > 0.0:
        fs = numerics.misidentification(fs,misid)
    
    return fs
