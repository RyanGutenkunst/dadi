import numpy as np
import dadi

## Simple SNM test. Check that all splits yield SNM.
#pts = 50
#xx = dadi.Numerics.default_grid(pts)
#phi = dadi.PhiManip.phi_1D(xx)
#fs1 = dadi.Spectrum.from_phi(phi, [10], (xx,))
#
#phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
#phi = dadi.PhiManip.phi_2D_to_3D_admix(phi, 0, xx,xx,xx)
#phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
#fs4 = dadi.Spectrum.from_phi(phi, [10,10,10,10], (xx,xx,xx,xx))
#
#for ii in range(4):
#    tomarg = list(range(4))
#    tomarg.remove(ii)
#    fsm = fs4.marginalize(tomarg)
#    print(np.allclose(fs1, fsm, rtol=1e-3, atol=1e-3))
#
## Simple SNM test with integration.
#pts = 30
#xx = dadi.Numerics.default_grid(pts)
#phi = dadi.PhiManip.phi_1D(xx)
#fs1 = dadi.Spectrum.from_phi(phi, [5], (xx,))
#
#phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
#phi = dadi.PhiManip.phi_2D_to_3D_admix(phi, 0, xx,xx,xx)
#phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
#phi = dadi.Integration.four_pops(phi, xx, T=0.1)
#fs4 = dadi.Spectrum.from_phi(phi, [5,5,5,5], (xx,xx,xx,xx))
#
#for ii in range(4):
#    tomarg = list(range(4))
#    tomarg.remove(ii)
#    fsm = fs4.marginalize(tomarg)
#    print(np.allclose(fs1, fsm, rtol=1e-2, atol=1e-2))

# 2D comparison tests
#pts = 20
#xx = dadi.Numerics.default_grid(pts)
#phi = dadi.PhiManip.phi_1D(xx)
#phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
#phi = dadi.Integration.two_pops(phi, xx, T=0.1, nu1=0.5, nu2=10, m12=2, m21=0.5, gamma1=-1, gamma2=1, h1=0.2, h2=0.9)
#fs2 = dadi.Spectrum.from_phi(phi, [5,5], (xx,xx))
#
#phi = dadi.PhiManip.phi_1D(xx)
#phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
#phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
#phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
#phi = dadi.Integration.four_pops(phi, xx, T=0.1, nu1=0.5, nu4=10, m14=2, m41=0.5, gamma1=-1, gamma4=1, h1=0.2, h4=0.9)
#fs4 = dadi.Spectrum.from_phi(phi, [5,5,5,5], (xx,xx,xx,xx))
#fsm = fs4.marginalize((1,2))
#print(np.allclose(fs2, fsm, rtol=1e-2, atol=1e-2))
#
#phi = dadi.PhiManip.phi_1D(xx)
#phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
#phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
#phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
#phi = dadi.Integration.four_pops(phi, xx, T=0.1, nu2=0.5, nu4=10, m24=2, m42=0.5, gamma2=-1, gamma4=1, h2=0.2, h4=0.7)
#fs4 = dadi.Spectrum.from_phi(phi, [5,5,5,5], (xx,xx,xx,xx))
#fsm = fs4.marginalize((0,2))
#print(np.allclose(fs2, fsm, rtol=1e-2, atol=1e-2))
#
#phi = dadi.PhiManip.phi_1D(xx)
#phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
#phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
#phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
#phi = dadi.Integration.four_pops(phi, xx, T=0.1, nu3=0.5, nu4=10, m34=2, m43=0.5, gamma3=-1, gamma4=1, h3=0.2, h4=0.7)
#fs4 = dadi.Spectrum.from_phi(phi, [5,5,5,5], (xx,xx,xx,xx))
#fsm = fs4.marginalize((0,1))
#print(np.allclose(fs2, fsm, rtol=1e-2, atol=1e-2))

#for ii in range(4):
#    tomarg = list(range(4))
#    tomarg.remove(ii)
#    fsm = fs4.marginalize(tomarg)
#    print(np.allclose(fs1, fsm, rtol=1e-2, atol=1e-2))

#
# admix_props test
#

#pts = 20
#xx = dadi.Numerics.default_grid(pts)
#phi = dadi.PhiManip.phi_1D(xx)
#phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
#phi = dadi.Integration.two_pops(phi, xx, T=0.1, nu1=0.5, nu2=10, m12=2, m21=0.5, gamma1=-1, gamma2=1, h1=0.2, h2=0.9)
#fs2 = dadi.Spectrum.from_phi(phi, [5,5], (xx,xx), admix_props=((0.3,0.7),(0.9,0.1)))
#
#phi = dadi.PhiManip.phi_1D(xx)
#phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
#phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
#phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
#phi = dadi.Integration.four_pops(phi, xx, T=0.1, nu3=0.5, nu4=10, m34=2, m43=0.5, gamma3=-1, gamma4=1, h3=0.2, h4=0.7)
#fs4 = dadi.Spectrum.from_phi(phi, [5,5,5,5], (xx,xx,xx,xx),
#                             admix_props=((1,0,0,0),(0,1,0,0),(0,0,0.3,0.7),(0,0,0.9,0.1)))
#fsm = fs4.marginalize((0,1))
#print(np.allclose(fs2, fsm, rtol=1e-2, atol=1e-2))

# Next phi_4D_admix methods

#pts = 20
#xx = dadi.Numerics.default_grid(pts)
#phi = dadi.PhiManip.phi_1D(xx)
#phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
#phi = dadi.Integration.two_pops(phi, xx, T=0.1, nu1=0.5, nu2=10, m12=2, m21=0.5, gamma1=-1, gamma2=1, h1=0.2, h2=0.9)
#phi = dadi.PhiManip.phi_2D_admix_1_into_2(phi,0.8,xx,xx)
#fs2 = dadi.Spectrum.from_phi(phi, [5,5], (xx,xx))
#
#phi = dadi.PhiManip.phi_1D(xx)
#phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
#phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
#phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
#phi = dadi.Integration.four_pops(phi, xx, T=0.1, nu3=0.5, nu4=10, m34=2, m43=0.5, gamma3=-1, gamma4=1, h3=0.2, h4=0.7)
#phi = dadi.PhiManip.phi_4D_admix_into_4(phi,0,0,0.8,xx,xx,xx,xx)
#fs4 = dadi.Spectrum.from_phi(phi, [5,5,5,5], (xx,xx,xx,xx))
#fsm = fs4.marginalize((0,1))
#print(np.allclose(fs2, fsm, rtol=1e-2, atol=1e-2))

#pts = 20
#xx = dadi.Numerics.default_grid(pts)
#phi = dadi.PhiManip.phi_1D(xx)
#phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
#phi = dadi.Integration.two_pops(phi, xx, T=0.1, nu1=0.5, nu2=10, m12=2, m21=0.5, gamma1=-1, gamma2=1, h1=0.2, h2=0.9)
#phi = dadi.PhiManip.phi_2D_admix_2_into_1(phi,0.8,xx,xx)
#fs2 = dadi.Spectrum.from_phi(phi, [5,5], (xx,xx))
#
#phi = dadi.PhiManip.phi_1D(xx)
#phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
#phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
#phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
#phi = dadi.Integration.four_pops(phi, xx, T=0.1, nu1=0.5, nu2=10, m12=2, m21=0.5, gamma1=-1, gamma2=1, h1=0.2, h2=0.7)
#phi = dadi.PhiManip.phi_4D_admix_into_1(phi,0.8,0,0,xx,xx,xx,xx)
#fs4 = dadi.Spectrum.from_phi(phi, [5,5,5,5], (xx,xx,xx,xx))
#fsm = fs4.marginalize((2,3))
#print(np.allclose(fs2, fsm, rtol=1e-2, atol=1e-2))

#pts = 20
#xx = dadi.Numerics.default_grid(pts)
#phi = dadi.PhiManip.phi_1D(xx)
#phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
#phi = dadi.Integration.two_pops(phi, xx, T=0.1, nu1=0.5, nu2=10, m12=2, m21=0.5, gamma1=-1, gamma2=1, h1=0.2, h2=0.9)
#phi = dadi.PhiManip.phi_2D_admix_1_into_2(phi,0.8,xx,xx)
#fs2 = dadi.Spectrum.from_phi(phi, [5,5], (xx,xx))
#
#phi = dadi.PhiManip.phi_1D(xx)
#phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
#phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
#phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
#phi = dadi.Integration.four_pops(phi, xx, T=0.1, nu1=0.5, nu2=10, m12=2, m21=0.5, gamma1=-1, gamma2=1, h1=0.2, h2=0.7)
#phi = dadi.PhiManip.phi_4D_admix_into_2(phi,0.8,0,0,xx,xx,xx,xx)
#fs4 = dadi.Spectrum.from_phi(phi, [5,5,5,5], (xx,xx,xx,xx))
#fsm = fs4.marginalize((2,3))
#print(np.allclose(fs2, fsm, rtol=1e-2, atol=1e-2))

pts = 20
xx = dadi.Numerics.default_grid(pts)
nu1_func = lambda t: 0.5 + 5*t
phi = dadi.PhiManip.phi_1D(xx)
phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
phi = dadi.Integration.two_pops(phi, xx, T=0.1, nu1=nu1_func, nu2=10, m12=2, m21=0.5, gamma1=-1, gamma2=1, h1=0.2, h2=0.9)
phi = dadi.PhiManip.phi_2D_admix_1_into_2(phi,0.8,xx,xx)
fs2 = dadi.Spectrum.from_phi(phi, [5,5], (xx,xx))

phi = dadi.PhiManip.phi_1D(xx)
phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
phi = dadi.Integration.four_pops(phi, xx, T=0.1, nu1=nu1_func, nu3=10, m13=2, m31=0.5, gamma1=-1, gamma3=1, h1=0.2, h3=0.7)
phi = dadi.PhiManip.phi_4D_admix_into_3(phi,0.8,0,0,xx,xx,xx,xx)
fs4 = dadi.Spectrum.from_phi(phi, [5,5,5,5], (xx,xx,xx,xx))
fsm = fs4.marginalize((1,3))
print(np.allclose(fs2, fsm, rtol=1e-2, atol=1e-2))