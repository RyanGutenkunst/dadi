"""
Bottleneck model with selection (sig1, sig2) on the two derived alleles, specifying length and severity of bottleneck, and length and strength of expansion, in units of 2Nref
"""

import numerics
import numpy as np
import integration
import dadi
from numpy import newaxis as nuax

def two_epoch(params, ns, pts, folded = False, misid = False):
    """
    params = [nu,T,sig1,sig2,theta1,theta2,misid]
    ns - number of samples
    """
    nu,T,sig1,sig2,theta1,theta2,p_mis = params
    # misid parameters are probability of misidentifying ancestral state for major and minor derived alleles (1 and 2 resp.)
    x = np.linspace(0,1,pts+1)
    sig1,sig2 = np.float(sig1),np.float(sig2)
    
    #y1 = integration.equilibrium_1D(x, 1.0, sig1, 1.0)
    #y2 = integration.equilibrium_1D(x, 1.0, sig2, 1.0)
    
    try: # open equilibrium distribution for given pts, sig1, and sig2
        open_file = np.load('Eq_pts' + str(pts) + '_sig1_' + str(sig1) + '_sig2_' + str(sig2) + '_theta1_' + str(theta1) + '_theta2_' + str(theta2) + '.npz')
        phi,y1,y2 = open_file['arr_0'], open_file['arr_1'], open_file['arr_2']
        open_file.close()
    except IOError: # if file doesn't exist, create equilibrium distr, and save it
        y1 = dadi.PhiManip.phi_1D(x,gamma=sig1)
        y2 = dadi.PhiManip.phi_1D(x,gamma=sig2)
        phi = np.zeros((len(x),len(x)))
        phi,y1,y2 = integration.equilibrium(phi, x, y1, y2, 1.0, sig1, sig2, theta1, theta2)
        np.savez('Eq_pts' + str(pts) + '_sig1_' + str(sig1) + '_sig2_' + str(sig2) + '_theta1_' + str(theta1) + '_theta2_' + str(theta2),phi,y1,y2)
    
    phi,y1,y2 = integration.advance(phi, x, T, y1, y2, nu, sig1, sig2, theta1, theta2)
    
    dx = numerics.grid_dx(x)
    DXX = dx[:,nuax]*dx[nuax,:]
    DXX[np.where(x[:,nuax] + x[nuax,:] == 1)] *= 1./2
    
    if not type(ns) == int:
        if len(ns) == 1:
            ns = int(ns)
        else:
            ns = int(ns[0])
    
    #fs = numerics.sample(phi, ns, x, DXX)
    fs = numerics.sample_cached(phi, ns, x, DXX)
    
    if folded == True:
        fs = numerics.fold(fs)
    
    if misid == True:
        fs = numerics.misidentification(fs,p_mis)
    
    fs = dadi.Spectrum(fs)
    fs.extrap_x = x[1]
    fs.mask[:,0] = True
    fs.mask[0,:] = True
    for ii in range(len(fs)):
        fs.mask[ii,ns-ii:] = True
    return fs


def bottleneck(params, ns, pts, folded = False, misid = False):
    nuB,nuF,TB,TF,sig1,sig2,theta1,theta2,p_mis,dt = params
    x = np.linspace(0,1,pts+1)
    sig1,sig2 = np.float(sig1),np.float(sig2)
    
    #y1 = integration.equilibrium_1D(x, 1.0, sig1, 1.0)
    #y2 = integration.equilibrium_1D(x, 1.0, sig2, 1.0)
    
    #try: # open equilibrium distribution for given pts, sig1, and sig2
    #    open_file = np.load('Eq_pts' + str(pts) + '_sig1_' + str(sig1) + '_sig2_' + str(sig2) + '_theta1_' + str(theta1) + '_theta2_' + str(theta2) + '.npz')
    #    phi,y1,y2 = open_file['arr_0'], open_file['arr_1'], open_file['arr_2']
    #    open_file.close()
    #except IOError: # if file doesn't exist, create equilibrium distr, and save it
    y1 = dadi.PhiManip.phi_1D(x,gamma=sig1)
    y2 = dadi.PhiManip.phi_1D(x,gamma=sig2)
    phi = np.zeros((len(x),len(x)))
    phi,y1,y2 = integration.equilibrium(phi, x, y1, y2, 1.0, sig1, sig2, theta1, theta2, dt)
    #    np.savez('Eq_pts' + str(pts) + '_sig1_' + str(sig1) + '_sig2_' + str(sig2) + '_theta1_' + str(theta1) + '_theta2_' + str(theta2),phi,y1,y2)

    phi,y1,y2 = integration.advance(phi, x, TB, y1, y2, nuB, sig1, sig2, theta1, theta2, dt)
    phi,y1,y2 = integration.advance(phi, x, TF, y1, y2, nuF, sig1, sig2, theta1, theta2, dt)

    dx = numerics.grid_dx(x)
    DXX = dx[:,nuax]*dx[nuax,:]
    DXX[np.where(x[:,nuax] + x[nuax,:] == 1)] *= 1./2
    
    if not type(ns) == int:
        if len(ns) == 1:
            ns = int(ns)
        else:
            ns = int(ns[0])
    
    #fs = numerics.sample(phi, ns, x, DXX)
    fs = numerics.sample_cached(phi, ns, x, DXX)
    
    if folded == True:
        fs = numerics.fold(fs)
    
    if misid == True:
        fs = numerics.misidentification(fs,p_mis)
    
    fs = dadi.Spectrum(fs)
    fs.extrap_x = x[1]
    fs.mask[:,0] = True
    fs.mask[0,:] = True
    for ii in range(len(fs)):
        fs.mask[ii,ns-ii:] = True
    return fs

