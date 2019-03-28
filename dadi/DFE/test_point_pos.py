import time
from dadi import Numerics, PhiManip, Integration, Spectrum
import Selection as S1d, Selection_2d as S2d
reload(S1d)
reload(S2d)

@Numerics.make_extrap_func
def split_mig_sel(params, ns, pts):
    nu1,nu2,T,m,gamma1,gamma2 = params

    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx, gamma=gamma1)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    phi = Integration.two_pops(phi, xx, T, nu1, nu2, m12=m, m21=m,
                               gamma1=gamma1, gamma2=gamma2)

    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs

def split_mig_single_sel(params, ns, pts):
    nu1,nu2,T,m,gamma = params
    return split_mig_sel((nu1,nu2,T,m,gamma,gamma), ns, pts)

demo_params = [1,0.2,0.2,0]
ns = [10,10]
pts_l = [30,40,50]
theta = 1e4

mu,sigma, rho = 10,4,0.9
ppos, gamma_pos = 0.1, 3.2
p2d = 0.4 # For mixture model, weight for 2D distribution

s1 = S1d.spectra(demo_params, ns, split_mig_single_sel, pts_l=pts_l, Npts=20,
                 int_bounds=(1e-4,50), echo=True, mp=True)

s2 = S2d.spectra2d(demo_params, ns, split_mig_sel, pts=pts_l, Npts=20,
                   int_bounds=(1e-4,50), echo=True, mp=True,
                   additional_gammas=[gamma_pos])

sel_dist1 = S1d.lognormal_dist
sel_dist2 = S2d.biv_lognorm_pdf

# First integration. Should add gamma_pos to cache in s1
start = time.time()
fs_single = s1.integrate_point_pos([mu,sigma,ppos,gamma_pos], sel_dist1,
                                   theta, split_mig_single_sel)
end = time.time()
print('First 1D integration: {0:.2f}s'.format(end-start))

# Second integration. Should be faster
start = time.time()
fs_single = s1.integrate_point_pos([mu,sigma,ppos,gamma_pos], sel_dist1,
                                   theta, split_mig_single_sel)
end = time.time()
print('Second 1D integration: {0:.2f}s'.format(end-start))

# Test that mixture model runs
start = time.time()
fs_mix = S2d.mixture_symmetric_point_pos([mu,sigma,rho,ppos,gamma_pos,p2d],
                                         ns, s1, s2, sel_dist1, sel_dist2,
                                         theta, split_mig_single_sel)
end = time.time()
print('Mixture model: {0:.2f}s'.format(end-start))
