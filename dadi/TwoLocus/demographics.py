import os
import dadi, numpy as np
from . import numerics, integration

# Cache equilibrium spectra in ~/.dadi/TwoLocus_cache by default
def set_cache_path(path='~/.dadi/TwoLocus_cache'):
    """
    Set directory in which demographic equilibrium phi spectra will be cached.

    The collection of cached spectra can get large, so it may be helpful to
    store them outside the user's home directory.
    """
    global cache_path
    cache_path = os.path.expanduser(path)
    if not os.path.isdir(cache_path):
        os.makedirs(cache_path)

cache_path=None
set_cache_path()

def equilibrium_phi(pts, rho, dt, gammaA=0, gammaB=0, hA=0, hB=0,
                    thetaA=1.0, thetaB=1.0):
    rho=float(rho); gammaA=float(gammaA); gammaB=float(gammaB);
    hA=float(hA); hB=float(hB); thetaA=float(thetaA); thetaB=float(thetaB)
    fname_template = 'phis_pts{0}_dt{1}_rho{2}_gammaA{3}_gammaB{4}_hA{5}_hB{6}_thetaA{7}_thetaB{8}.npz'
    fname = fname_template.format(pts,rho,dt,gammaA,gammaB,hA,hB,thetaA,thetaB)
    fname = os.path.join(cache_path, fname)
    try:
        fid = np.load(fname)
        yA,yB,phi = fid['arr_0'], fid['arr_1'], fid['arr_2']
        fid.close()
    except IOError:
        print('Calculating equilibrium phi from scratch.')
        x = numerics.grid(pts)

        yA = dadi.PhiManip.phi_1D(x, gamma = gammaA, h = hA, theta0 = thetaA)
        yB = dadi.PhiManip.phi_1D(x, gamma = gammaB, h = hB, theta0 = thetaB)

        phi = np.zeros((len(x),len(x),len(x)))
        yA, yB, phi = integration.advance(phi, x, 20., yA, yB, 1.0,
                                          gammaA, gammaB, hA, hB, rho,
                                          thetaA, thetaB, dt)
        np.savez(fname, yA, yB, phi)
    return yA, yB, phi

def equilibrium(pts, ns, rho=0.0, dt=0.005, gammaA=0.0, gammaB=0.0,
                 hA=0.5, hB=0.5, thetaA=1.0, thetaB=1.0):
    x = numerics.grid(pts)
    yA, yB, phi = equilibrium_phi(pts, rho, dt, gammaA, gammaB, hA, hB,
                                  thetaA, thetaB)

    fs = numerics.sample_cached(phi, ns, x)
    return fs

def two_epoch(params, pts, ns, rho=0.0, dt=0.005, gammaA=0.0, gammaB=0.0,
              hA=0.5, hB=0.5, thetaA=1.0, thetaB=1.0):
    nu,T = params

    x = numerics.grid(pts)
    yA, yB, phi = equilibrium_phi(pts, rho, dt, gammaA, gammaB, hA, hB,
                                  thetaA, thetaB)

    yA,yB,phi = integration.advance(phi, x, T, yA, yB, nu=nu,
                                    gammaA=gammaA, gammaB=gammaB,
                                    hA=hA, hB=hB, rho=rho,
                                    thetaA=thetaA, thetaB=thetaB, dt=dt)

    fs = numerics.sample_cached(phi, ns, x)
    return fs

def three_epoch(params, pts, ns, rho=0.0, dt=0.005, gammaA=0.0, gammaB=0.0,
                hA=0.5, hB=0.5, thetaA=1.0, thetaB=1.0):
    nu1,nu2,T1,T2 = params
    x = numerics.grid(pts)

    yA, yB, phi = equilibrium_phi(pts, rho, dt, gammaA, gammaB, hA, hB,
                                  thetaA, thetaB)

    yA,yB,phi = integration.advance(phi, x, T1, yA, yB, nu=nu1,
                                    gammaA=gammaA, gammaB=gammaB,
                                    hA=hA, hB=hB, rho=rho,
                                    thetaA=thetaA, thetaB=thetaB, dt=dt)

    yA,yB,phi = integration.advance(phi, x, T2, yA, yB, nu=nu2,
                                    gammaA=gammaA, gammaB=gammaB,
                                    hA=hA, hB=hB, rho=rho,
                                    thetaA=thetaA, thetaB=thetaB, dt=dt)

    fs = numerics.sample_cached(phi, ns, x)
    return fs
