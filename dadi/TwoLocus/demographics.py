import dadi, numpy as np
import numerics, integration
import copy

# I want to cache equilibrium spectra within the .../dadi/TwoLocus/cached_eq directory
import os
cache_path = os.path.abspath(dadi.__file__)
cache_path = cache_path.split('__')[0]
cache_path += 'TwoLocus/cached_eq/'


def equilibrium(numpts, ns, rho=0.0, dt=0.005, gammaA = 0.0, gammaB = 0.0, hA = 0.5, hB = 0.5):
    rho = float(rho); gammaA = float(gammaA); gammaB = float(gammaB)
    eq_fname = cache_path + 'phis_pts{0}_dt{1}_rho{2}_gammaA{3}_gammaB{4}.npz'.format(numpts,dt,rho,gammaA,gammaB)
    x = numerics.grid(numpts)
    dx = numerics.grid_dx(x)
    U01 = numerics.domain(x)
    dx3 = numerics.grid_dx3(x,dx)
    try:
        open_file = np.load(eq_fname)
        #print 'using cached spectra from equilibrium'
        yA,yB,phi = open_file['arr_0'], open_file['arr_1'], open_file['arr_2']
        open_file.close()
    except IOError:
        print 'calculating equilibrium from scratch'
        yA = dadi.PhiManip.phi_1D(x,gamma=gammaA)
        yB = dadi.PhiManip.phi_1D(x,gamma=gammaB)
        phi = np.zeros((len(x),len(x),len(x)))
        yA,yB,phi = integration.advance(phi, x, 20., yA, yB, gammaA=gammaA, gammaB=gammaB, hA=hA, hB=hB, rho=rho, thetaA=1., thetaB=1., dt=dt)
        np.savez(eq_fname,yA,yB,phi)

    fs = numerics.sample_cached(phi, ns, x, dx3)
    return fs

def two_epoch(params, numpts, ns, rho=0.0, dt=0.005, gammaA = 0.0, gammaB = 0.0, hA = 0.5, hB = 0.5):
    rho = float(rho); gammaA = float(gammaA); gammaB = float(gammaB)
    nu,T = params
    x = numerics.grid(numpts)
    dx = numerics.grid_dx(x)
    U01 = numerics.domain(x)
    dx3 = numerics.grid_dx3(x,dx)

    eq_fname = cache_path + 'phis_pts{0}_dt{1}_rho{2}_gammaA{3}_gammaB{4}.npz'.format(numpts,dt,rho,gammaA,gammaB)
    try:
        open_file = np.load(eq_fname)
        #print 'using cached spectra from equilibrium'
        yA,yB,phi = open_file['arr_0'], open_file['arr_1'], open_file['arr_2']
        open_file.close()
    except IOError:
        print 'calculating equilibrium from scratch for [{0},{1}]'.format(numpts,dt)
        yA = dadi.PhiManip.phi_1D(x)  ## account for gamma!!!
        yB = dadi.PhiManip.phi_1D(x)  ## account for gamma!!!
        phi = np.zeros((len(x),len(x),len(x)))
        yA,yB,phi = integration.advance(phi, x, 20., yA, yB, gammaA=gammaA, gammaB=gammaB, hA=hA, hB=hB, rho=rho, thetaA=1., thetaB=1., dt=dt)
        np.savez(eq_fname,yA,yB,phi)
    
    if T > 0:
        yA,yB,phi = integration.advance(phi, x, T, yA, yB, nu=nu, gammaA=gammaA, gammaB=gammaB, hA=hA, hB=hB, rho=rho, thetaA=1.0, thetaB=1.0, dt=dt)
    
    fs = numerics.sample_cached(phi, ns, x, dx3)
    return fs



def three_epoch(params, numpts, ns, rho=0.0, dt=0.005, gammaA = 0.0, gammaB = 0.0, hA = 0.5, hB = 0.5):
    rho = float(rho); gammaA = float(gammaA); gammaB = float(gammaB)
    nu1,nu2,T1,T2 = params
    x = numerics.grid(numpts)
    dx = numerics.grid_dx(x)
    U01 = numerics.domain(x)
    dx3 = numerics.grid_dx3(x,dx)

    eq_fname = cache_path + 'phis_pts{0}_dt{1}_rho{2}_gammaA{3}_gammaB{4}.npz'.format(numpts,dt,rho,gammaA,gammaB)
    try:
        open_file = np.load(eq_fname)
        #print 'using cached spectra from equilibrium'
        yA,yB,phi = open_file['arr_0'], open_file['arr_1'], open_file['arr_2']
        open_file.close()
    except IOError:
        print 'calculating equilibrium from scratch'
        yA = dadi.PhiManip.phi_1D(x)  ## account for gamma!!!
        yB = dadi.PhiManip.phi_1D(x)  ## account for gamma!!!
        phi = np.zeros((len(x),len(x),len(x)))
        yA,yB,phi = integration.advance(phi, x, 20., yA, yB, gammaA=gammaA, gammaB=gammaB, hA=hA, hB=hB, rho=rho, thetaA=1.0, thetaB=1.0, dt=dt)
        np.savez(eq_fname,yA,yB,phi)
    
    if T1 > 0:
        yA,yB,phi = integration.advance(phi, x, T1, yA, yB, nu=nu1, gammaA=gammaA, gammaB=gammaB, hA=hA, hB=hB, rho=rho, thetaA=1.0, thetaB=1.0, dt=dt)
    
    if T2 > 0:
        yA,yB,phi = integration.advance(phi, x, T2, yA, yB, nu=nu2, gammaA=gammaA, gammaB=gammaB, hA=hA, hB=hB, rho=rho, thetaA=1.0, thetaB=1.0, dt=dt)
    
    fs = numerics.sample_cached(phi, ns, x, dx3)
    return fs