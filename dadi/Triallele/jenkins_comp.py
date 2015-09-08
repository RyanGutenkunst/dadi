"""
Comparison to Jenkins et al (2014) neutral triallelic freq spectrum with arbitrary demography. Require moments from trees, which we generated using ms (10,000 trees created, using tree_script.sh).
We compare with ns = 20, with demography nu = 2.0, tau = .1
These parameters in ms are nu_ms = 0.5, tau = 0.025
"""

import dadi
import numpy as np, scipy, math, scipy.io
import matplotlib.pylab as plt
from scipy.special import binom as binom

nu, T = 2.0, 0.1 # instantaneous population size change (doubled in size) 0.1 time units (in 2Ne generations) ago
sig1 = 0.0
sig2 = 0.0
theta1 = 1.
theta2 = 1.
misid = 0.0 # no misidentification

grid_pts = [60,80,100] # evaluate over these grid points, then extrapolate to $\Delta = 0$
ns = 20 # number of observed samples

dt0 = 0.01 # time step of integration - should automate?
dt1 = 0.001
dt2 = 0.0001

# for dt0
params = [nu,T,sig1,sig2,theta1,theta2,misid,dt0]
fs00 = dadi.Triallele.demographics.two_epoch(params,ns,grid_pts[0], folded = False, misid = False)
fs10 = dadi.Triallele.demographics.two_epoch(params,ns,grid_pts[1], folded = False, misid = False)
fs20 = dadi.Triallele.demographics.two_epoch(params,ns,grid_pts[2], folded = False, misid = False)

# for dt1
params = [nu,T,sig1,sig2,theta1,theta2,misid,dt1]
fs01 = dadi.Triallele.demographics.two_epoch(params,ns,grid_pts[0], folded = False, misid = False)
fs11 = dadi.Triallele.demographics.two_epoch(params,ns,grid_pts[1], folded = False, misid = False)
fs21 = dadi.Triallele.demographics.two_epoch(params,ns,grid_pts[2], folded = False, misid = False)

# for dt2
params = [nu,T,sig1,sig2,theta1,theta2,misid,dt2]
fs02 = dadi.Triallele.demographics.two_epoch(params,ns,grid_pts[0], folded = False, misid = False)
fs12 = dadi.Triallele.demographics.two_epoch(params,ns,grid_pts[1], folded = False, misid = False)
fs22 = dadi.Triallele.demographics.two_epoch(params,ns,grid_pts[2], folded = False, misid = False)

fs0 = dadi.Numerics.quadratic_extrap((fs00,fs10,fs20),(fs00.extrap_x,fs10.extrap_x,fs20.extrap_x))
fs1 = dadi.Numerics.quadratic_extrap((fs01,fs11,fs21),(fs01.extrap_x,fs11.extrap_x,fs21.extrap_x))
fs2 = dadi.Numerics.quadratic_extrap((fs02,fs12,fs22),(fs02.extrap_x,fs12.extrap_x,fs22.extrap_x))

fs = dadi.Numerics.quadratic_extrap((fs0,fs1,fs2),(dt0,dt1,dt2))

### Jenkins calculations

Pab = .5
Pac = .5
Pbc = .5
Pcb = .5

# calculate moments from ms output of tree file
def moments(filename,samples):
    """
    M = first moments, counting down from number of samples to 2
    MM = second moments
    """
    f = open(filename,'r')
    lines = f.readlines()
    T = np.zeros((len(lines),len(str.split(lines[0]))))
    c = 0
    for l in lines:
        m = str.split(l)
        for n in range(len(m)):
            T[c,n] = np.array(m[n]).astype(np.float)
        c += 1

    T = 2*T # since we want to scale by 2N, not 4N as in ms

    M = np.zeros(samples-1)
    for ii in range(len(T[0])):
        M[ii] = np.mean(T[:,ii])
        #print "E[T" + str(len(T[0]) + 1 - ii) + "] = " + str(np.mean(T[:,ii]))

    MM = np.zeros((samples-1,samples-1))
    for ii in range(len(T[0])):
        for jj in range(len(T[0])):
            MM[ii,jj] = np.mean(T[:,ii]*T[:,jj])
            #print "E[T" + str(len(T[0]) + 1 - ii) + " T" + str(len(T[0]) + 1 - jj) + "] = " + str(np.mean(T[:,ii]*T[:,jj]))

    f.close()
    return M,MM

def ind(k,na,nb):
    return 1 if k <= na+nb+1 else 0

def delta(j,k):
    return 1 if j == k else 0

def C(na,nb,nc,j,k):
    sum = 0
    n = na + nb + nc
    for l in range(j-1,k-1):
        sum += binom(na-1,l-1) * binom(nb-1,k-l-2) * binom(k-j,k-1-l) / binom(n-1,k-1) / binom(k-1,k-l) * j*(j-1)
    return sum

def D(j,k):
    return j * ( k * binom(k-2,j-1) - (j-1) * binom(k-1,j) ) / binom(k-1,j-1)

def F(na,nb,nc,j,k):
    sum = 0
    n = na + nb + nc
    for l in range(max(j-2,1),k-1):
        sum += binom(na-1,l-1) * binom(nb-1,k-l-2) * binom(k-j,k-2-l) / binom(n-1,k-1) / binom(k-1,l+1) * j*(j-1)/(1+delta(j,k))
    return sum

def G(j,k):
    return (k*(j-1) - 2*delta(j,2)/(k-1)) * 1/(1+delta(j,k))

def gamma(na,nb,nc,j,k):
    return (Pab*Pbc*C(na,nb,nc,j,k) + Pab*Pac*F(na,nb,nc,j,k)) * ind(k,na,nb) + (Pac*Pcb*C(na,nc,nb,j,k) + Pab*Pac*F(na,nc,nb,j,k)) * ind(k,na,nc)

def kappa(j,k):
    return (1./2) * D(j,k) + (1./2) * G(j,k)
    
def exp_eq(j,k,delta): #expectation of second moment
    return (1+delta(j,k)) / binom(j,2) / binom(k,2)

def exp(M,MM,j,k): # expectation of second moment E[TjTk]
    return MM[j,k]

M,MM = moments('intervals_nu2.0_tau.1_ns20.txt',ns)
Phi = np.zeros((ns+1,ns+1))

for ii in range(ns+1)[1:]: ## nb
    for jj in range(ns+1)[1:]: ## nc
        if ii+jj < ns:
            numer = 0
            denom = 0
            for k in range(3,ns+1):
                for j in range(2,k+1):
                    numer += gamma(ns-ii-jj,ii,jj,j,k) * exp(M,MM,ns-j,ns-k) #* exp_eq(j,k)
                    denom += kappa(j,k) * exp(M,MM,ns-j,ns-k) #* exp_eq(j,k)
            Phi[ii,jj] = numer/denom

Phi = dadi.Spectrum(Phi)
Phi.mask[0,:] = True
Phi.mask[:,0] = True
for ii in range(len(Phi)):
    Phi.mask[ii,ns-ii:] = True

###
cmap = plt.get_cmap('cubehelix_r')

# plot differences
fs = dadi.Triallele.numerics.fold(fs)/np.sum(fs)
Phi = dadi.Triallele.numerics.fold(Phi)/np.sum(Phi)

fig = plt.figure(1001,figsize=(8,6),dpi=150)
ax1 = plt.subplot(3,2,1,aspect='equal')
dadi.Plotting.plot_single_2d_sfs(np.transpose(fs),pop_ids=('Minor DAF','Major DAF'),cmap=cmap)
ax1.set_xlim([0,20])
ax1.set_ylim([0,12])
ax1.set_title('Diffusion')

ax2 = plt.subplot(3,2,3,aspect='equal')
dadi.Plotting.plot_single_2d_sfs(np.transpose(Phi),pop_ids=('Minor DAF','Major DAF'),cmap=cmap)
ax2.set_xlim([0,20])
ax2.set_ylim([0,12])
ax2.set_title('Coalescent')

ax3 = plt.subplot(3,2,5,aspect='equal')
resid = dadi.Inference.linear_Poisson_residual(fs, Phi)
plt.pcolor(np.transpose(resid),cmap=plt.cm.RdBu_r, vmin=-abs(resid).max(), vmax=abs(resid).max())
plt.colorbar()
ax3.set_xlim([0,20])
ax3.set_ylim([0,12])
ax3.set_title('Residual')

## compare equilibrium exact frequencies
ax4 = plt.subplot(3,2,2,aspect='equal')

fs60 = dadi.Triallele.numerics.sample_cached( dadi.Triallele.integration.equilibrium_neutral_exact(np.linspace(0,1,61)) , ns , np.linspace(0,1,61), dadi.Triallele.numerics.grid_dx_2d(np.linspace(0,1,61),dadi.Triallele.numerics.grid_dx(np.linspace(0,1,61))) )

fs80 = dadi.Triallele.numerics.sample_cached( dadi.Triallele.integration.equilibrium_neutral_exact(np.linspace(0,1,81)) , ns , np.linspace(0,1,81), dadi.Triallele.numerics.grid_dx_2d(np.linspace(0,1,81),dadi.Triallele.numerics.grid_dx(np.linspace(0,1,81))) )

fs100 = dadi.Triallele.numerics.sample_cached( dadi.Triallele.integration.equilibrium_neutral_exact(np.linspace(0,1,101)) , ns , np.linspace(0,1,101), dadi.Triallele.numerics.grid_dx_2d(np.linspace(0,1,101),dadi.Triallele.numerics.grid_dx(np.linspace(0,1,101))) )

fseq = dadi.Numerics.quadratic_extrap((fs60,fs80,fs100),(fs60.extrap_x,fs80.extrap_x,fs100.extrap_x))

fseq /= np.sum(fseq)

fseq = dadi.Triallele.numerics.fold(fseq)

dadi.Plotting.plot_single_2d_sfs(np.transpose(fseq),pop_ids=('Minor DAF','Major DAF'),cmap=cmap)
ax4.set_xlim([0,20])
ax4.set_ylim([0,12])
ax4.set_title('Extrapolation on $\Delta x$')

ax5 = plt.subplot(3,2,4,aspect='equal')

Phieq = np.zeros((ns+1,ns+1))

def d(na,nb,nc,n):
    if nc > 1:
        return 1./((na+nb)*(na+nb-1))*(1 + 1.*n/nc - 2.*n*(np.sum(1./np.linspace(1,n,n))-np.sum(1./np.linspace(1,nc-1,nc-1)))/(na+nb+1))
    else:
        return 1./((na+nb)*(na+nb-1))*(1 + 1.*n/nc - 2.*n*(np.sum(1./np.linspace(1,n,n)))/(na+nb+1))

for ii in range(ns+1)[1:]: # ii is type b
    for jj in range(ns+1)[1:]: # jj in type c
        if ii+jj < ns:
            C = ( 1./2 ) * (np.sum(1./np.linspace(1,ns,ns)) + 1./ns - 2) + ( 1./2 ) * ( np.sum(1./np.linspace(1,ns,ns))**2/2 - np.sum(1./np.linspace(1,ns,ns)**2)/2 - np.sum(1./np.linspace(1,ns,ns)) - 1./ns + 2)
            Phieq[ii,jj] = 1./C * ( Pab*Pbc*d(ns-ii-jj,ii,jj,ns) + Pac*Pcb*d(ns-ii-jj,jj,ii,ns) + Pab*Pac*(1./(ii*jj) - d(ns-ii-jj,ii,jj,ns) - d(ns-ii-jj,jj,ii,ns)) )

Phieq = Phieq/np.sum(Phieq)
Phieq = dadi.Spectrum(Phieq)
Phieq.mask[0,:] = True
Phieq.mask[:,0] = True
for ii in range(len(Phieq)):
    Phieq.mask[ii,ns-ii:] = True
    
Phieq = dadi.Triallele.numerics.fold(Phieq)

dadi.Plotting.plot_single_2d_sfs(np.transpose(Phieq),pop_ids=('Minor DAF','Major DAF'),cmap=cmap)
ax5.set_xlim([0,20])
ax5.set_ylim([0,12])
ax5.set_title('Exact')

ax6 = plt.subplot(3,2,6,aspect='equal')
resideq = dadi.Inference.linear_Poisson_residual(fseq, Phieq)
plt.pcolor(np.transpose(resideq),cmap=plt.cm.RdBu_r, vmin=-abs(resideq).max(), vmax=abs(resideq).max())
plt.colorbar()
ax6.set_xlim([0,20])
ax6.set_ylim([0,12])
ax6.set_title('Residual')


fig.tight_layout()
plt.savefig('jenkins_comparison.pdf')
plt.show()
