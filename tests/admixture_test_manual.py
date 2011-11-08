"""
Test script for validating performance of different simulation techniques
when admixture is present.
"""
import cPickle, os, sys, time
import numpy
from numpy import array
import matplotlib.pyplot as pyplot
import dadi

REFRESH_DADI = True

try:
    dadi_cache = cPickle.load(file('dadi_cache.bp'))
    ms_cache = cPickle.load(file('ms_cache.bp'))
except IOError:
    dadi_cache = {}
    ms_cache = {}


def demo_func(params, ns, pts, crwd=8, method='PhiManip'):
    nuC, nu1, nu2, nu1B, nu1F, Tc, Ts, Tb, m12, m21, f, Tf = params
    xx = yy = dadi.Numerics.default_grid(pts, crwd=crwd)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.Integration.one_pop(phi, xx, Tc, nuC)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1, nu2, m12, m21)
    nu_func = lambda t: nu1B * numpy.exp(numpy.log(nu1F/nu1B)*t/Tb)
    phi = dadi.Integration.two_pops(phi, xx, Tb, nu_func, nu2, m12, m21)

    if method == 'PhiManip':
        phi = dadi.PhiManip.phi_2D_admix_2_into_1(phi, f, xx, yy)
        phi = dadi.Integration.two_pops(phi, xx, Tf, nu_func(Tb), nu2, m12, m21)
        fs = dadi.Spectrum.from_phi(phi, ns, (xx,xx))
    elif method == 'Direct':
        fs = dadi.Spectrum.from_phi(phi, ns, (xx, yy), 
                                    admix_props=((1-f, f), (0, 1)))

    return fs

def ms_corefunc((nuC, nu1, nu2, nu1B, nu1F, Tc, Ts, Tb, m12, m21, f, Tf)):
	alpha1 = numpy.log(nu1F/nu1B)/Tb
	
	command = '-n 1 %(nu1F)f -n 2 %(nu2)f '\
		'-es %(Tf)f 1 %(admix)f -ej %(Tf)f 3 2 '\
		'-ma x %(m12)f %(m21)f x '\
		'-eg %(Tf)f 1 %(alpha1)f '\
		'-eg %(Tb)f 1 0 -en %(Tb)f 1 %(nu1)f '\
		'-en %(Ts)f 1 %(nuC)f -ej %(Ts)f 2 1 '\
		'-en %(Tsum)f 1 1'

        sub_dict = { 'nuC':nuC, 'nu1':nu1, 'nu2':nu2, 'nu1B':nu1B, 'nu1F':nu1F, 'Tf':Tf/2, 'Tb':(Tb+Tf)/2, 'Ts':((Tb+Ts+Tf)/2), 'Tsum':((Tc+Ts+Tb+Tf)/2), 'alpha1':2*alpha1, 'm12':2*m12, 'm21':2*m21, 'admix':1-f }
	return command % sub_dict

def cached_demo_func(params, ns, pts, crwd=8, method='PhiManip'):
    key = (tuple(params), ns, pts, crwd, method)
    if REFRESH_DADI or not dadi_cache.has_key(key):
        start = time.time()
        dadi_cache[key] = demo_func(params, ns, pts, crwd=crwd, method=method)
        print 'Time for dadi, pts:', time.time() - start, pts
        sys.stdout.flush()
    return dadi_cache[key]

def cached_ms_run(params, ns, theta_tot):
    key = (tuple(params), ns)
    theta_cached, result = ms_cache.get(key, (0,None))
    if theta_cached < theta_tot:
        ms_core = ms_corefunc(params)
        mscommand = dadi.Misc.ms_command(1, ns, ms_core, theta_tot) 
        start = time.time()
        result = dadi.Spectrum.from_ms_file(os.popen(mscommand), 
                                            pop_ids=['BIA','YRI'],
                                            average=False)
        print 'Time for ms:', time.time() - start
        sys.stdout.flush()
        ms_cache[key] = (theta_tot, result)
    else:
        theta_tot = theta_cached
    return result, theta_tot

ns = (8,18)
func_ex = dadi.Numerics.make_extrap_log_func(cached_demo_func)

params = array([0.17, 0.44, 1.11, 3.04, 0.26, 0.14, 0.57, 0.05, 7.04, 1.19, 
                0.27, 0])

###############################################################################
#
# Plot for comparison with ms
#
###############################################################################

crwd = 8
method = 'Direct'

pts_l = [30,40,50]

model = func_ex(params, ns, pts_l, crwd=crwd, method=method)
msdata, theta_tot = cached_ms_run(params, ns, 1e5)

model *= theta_tot
ll_model = dadi.Inference.ll(model, msdata)
theta = dadi.Inference.optimal_sfs_scaling(model, msdata)
print 'LL:', ll_model
print 'theta scaling:', theta

dadi.Plotting.plot_2d_comp_Poisson(model, msdata, vmin=1, fig_num=1)
fig = pyplot.gcf()
fig.axes[0].set_title('ms')
fig.axes[1].set_title('dadi')

###############################################################################
#
# Plot for smoothness of extrapolation
#
###############################################################################

bia_ii, yri_jj = 2, 0

all_pts = range(40,121,10)
all_fs = func_ex(params, ns, all_pts, crwd=crwd, method=method, 
                 no_extrap=True)
all_dx = [fs.extrap_x for fs in all_fs]
all_fs_array = numpy.array(all_fs)

fig = pyplot.figure(109)
fig.clear()
ax = fig.add_subplot(1,1,1)
ax.plot(all_dx, theta_tot*all_fs_array[:,bia_ii,yri_jj], '-o')
msval = msdata[bia_ii, yri_jj]
ax.errorbar([0], [msval], yerr=[numpy.sqrt(msval)], fmt='-or', zorder=-10)
ax.plot([0], [model[bia_ii,yri_jj]], 'og', zorder=-10)
ax.axis(xmin=-1e-5)
ax.set_title('Entry [bia={0:d}, yri={1:d}]'.format(bia_ii, yri_jj))

lls = [dadi.Inference.ll(theta_tot*m, msdata) for m in all_fs]
for pts, ll in zip(all_pts, lls):
    print pts, ll

###############################################################################
#
# Clean up
#
###############################################################################

fid = file('ms_cache.bp', 'w')
cPickle.dump(ms_cache, fid, 2)
fid.close()
fid = file('dadi_cache.bp', 'w')
cPickle.dump(dadi_cache, fid, 2)
fid.close()

pyplot.show()
