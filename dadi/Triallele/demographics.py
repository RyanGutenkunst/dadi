"""
Equilibrium, two epoch, and three epoch (bottleneck) model with selection (sig1, sig2) on the two derived alleles
"""

import numerics
import numpy as np
import integration
import dadi
from numpy import newaxis as nuax

def equilibrium(params, ns, pts, folded = False, misid = False):
    """
    Integrate the density function to equilibrium
    params = [sig1,sig2,theta1,theta2,misid]
    
    """
    sig1,sig2,theta1,theta2,misid,dt = params
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
    DXX = numerics.grid_dx_2d(x,dx)
    
    if not type(ns) == int:
        if len(ns) == 1:
            ns = int(ns)
        else:
            ns = int(ns[0])
    
    fs = numerics.sample_cached(phi, ns, x, DXX)
    
    if folded == True:
        fs = numerics.fold(fs)
    
    if misid == True:
        fs = numerics.misidentification(fs,misid)
    
    fs = dadi.Spectrum(fs)
    fs.extrap_x = x[1]
    # mask out non-triallelic and non-interior entries
    fs.mask[:,0] = True
    fs.mask[0,:] = True
    for ii in range(len(fs)):
        fs.mask[ii,ns-ii:] = True
    
    return fs
    
def two_epoch(params, ns, pts, folded = False, misid = False):
    """
    params = [nu,T,sig1,sig2,theta1,theta2,misid,dt]
    """
    nu,T,sig1,sig2,theta1,theta2,misid,dt = params

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
    
    phi,y1,y2 = integration.advance(phi, x, T, y1, y2, nu, sig1, sig2, theta1, theta2, dt)
    
    dx = numerics.grid_dx(x)
    DXX = numerics.grid_dx_2d(x,dx)
    
    if not type(ns) == int:
        if len(ns) == 1:
            ns = int(ns)
        else:
            ns = int(ns[0])
    
    fs = numerics.sample_cached(phi, ns, x, DXX)
    
    if folded == True:
        fs = numerics.fold(fs)
    
    if misid == True:
        fs = numerics.misidentification(fs,misid)
    
    fs = dadi.Spectrum(fs)
    fs.extrap_x = x[1]
    # mask out non-triallelic and non-interior entries
    fs.mask[:,0] = True
    fs.mask[0,:] = True
    for ii in range(len(fs)):
        fs.mask[ii,ns-ii:] = True
    
    return fs


def three_epoch(params, ns, pts, folded = False, misid = False):
    nuB,nuF,TB,TF,sig1,sig2,theta1,theta2,misid,dt = params
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
    
    phi,y1,y2 = integration.advance(phi, x, TB, y1, y2, nuB, sig1, sig2, theta1, theta2, dt)
    phi,y1,y2 = integration.advance(phi, x, TF, y1, y2, nuF, sig1, sig2, theta1, theta2, dt)

    dx = numerics.grid_dx(x)
    DXX = numerics.grid_dx_2d(x,dx)
    
    if not type(ns) == int:
        if len(ns) == 1:
            ns = int(ns)
        else:
            ns = int(ns[0])
    
    fs = numerics.sample_cached(phi, ns, x, DXX)
    
    if folded == True:
        fs = numerics.fold(fs)
    
    if misid == True:
        fs = numerics.misidentification(fs,misid)
    
    fs = dadi.Spectrum(fs)
    fs.extrap_x = x[1]
    # mask out non-triallelic and non-interior entries
    fs.mask[:,0] = True
    fs.mask[0,:] = True
    for ii in range(len(fs)):
        fs.mask[ii,ns-ii:] = True
    
    return fs

