import pickle, random
import numpy as np
import matplotlib.pyplot as plt

import dadi
# Pulls in Cache1D, Cache2D, mixture, mixture_symmetric_point_pos
#          PDFs, DemogSelModels, Plotting
from dadi.DFE import *

# For multiprocessing to work on Windows, all script code must be wrapped
# in this block. If you're not on Windows, feel free to remove this if statement.
if __name__ == '__main__':
    # Seed random number generator, so example is reproducible
    np.random.seed(1398238)
    
    #
    # Plotting a joint DFE
    #
    sel_dist = PDFs.biv_lognormal
    params = [0.5,-0.5,0.5,1,-0.8]
    gammax, gammay = -np.logspace(-2, 1, 20), -np.logspace(-1, 2, 30)
    
    fig = plt.figure(137, figsize=(4,3), dpi=150)
    fig.clear()
    ax = fig.add_subplot(1,1,1)
    Plotting.plot_biv_dfe(gammax, gammay, sel_dist, params, logweight=True, ax=ax)
    fig.tight_layout()
    
    # With positive selection
    params = [0.5,-0.5,0.5,1,0.0,0.3,3,0.3,4]
    fig = Plotting.plot_biv_point_pos_dfe(gammax, gammay, sel_dist, params,
                                          fignum=23, rho=params[4])
    
    #
    # Full test of optimization machinery. 
    # Considering only a narrow range of gammas, so integration is faster.
    #
    demo_params = [0.5,2,0.5,0.1,0,0]
    ns = [8, 12]
    pts_l = [60, 80, 100]
    func_ex = DemogSelModels.IM_sel
    # Check whether we already have a chached set of 2d spectra. If not
    # generate them.
    try:
        s2 = pickle.load(open('test.spectra2d.bpkl', 'rb'))
    except IOError:
        s2 = Cache2D(demo_params, ns, func_ex, pts=pts_l, gamma_pts=100,
                     gamma_bounds=(1e-2, 10), verbose=True, cpus=4,
                     additional_gammas=[1.2, 4.3])
        # Save spectra2d object
        fid = open('test.spectra2d.bpkl', 'wb')
        pickle.dump(s2, fid, protocol=2)
        fid.close()

        ## Cache generation can be very computationally expensive, even
        ## with multiprocessing. If you need to split the work across multiple
        ## compute nodes, use the split_jobs and this_job_id arguments, then
        ## merge the partial caches.
        ## In this example, the 3 partial caches could be generate on independent
        ## nodes, then s2a,s2b,s2c would be saved to separate files, then loaded
        ## and combined later.
        #s2a = Cache2D(demo_params, ns, func_ex, pts=pts_l, gamma_pts=100,
        #        gamma_bounds=(1e-2, 10), verbose=True, cpus=4,
        #        additional_gammas=[1.2, 4.3], split_jobs=3, this_job_id=0)
        #s2b = Cache2D(demo_params, ns, func_ex, pts=pts_l, gamma_pts=100,
        #        gamma_bounds=(1e-2, 10), verbose=True, cpus=4,
        #        additional_gammas=[1.2, 4.3], split_jobs=3, this_job_id=1)
        #s2c = Cache2D(demo_params, ns, func_ex, pts=pts_l, gamma_pts=100,
        #        gamma_bounds=(1e-2, 10), verbose=True, cpus=4,
        #        additional_gammas=[1.2, 4.3], split_jobs=3, this_job_id=2)
        #s2 = Cache2D.merge([s2a, s2b, s2c])
    
    # Generate test data set to fit
    input_params, theta = [0.5,0.5,-0.8], 1e5
    sel_dist = PDFs.biv_lognormal
    # Expected sfs
    target = s2.integrate(input_params, None, sel_dist, theta, None)
    # Get data with Poisson variance around expectation
    data = target.sample()
    
    # Parameters are mean, variance, and correlation coefficient
    p0 = [0,1.,0.8]
    popt,llopt = dadi.Inference.opt(p0, data, s2.integrate, pts=None,
                                   func_args=[sel_dist, theta],
                                   lower_bound=[None,0,-1],
                                   upper_bound=[None,None,1],
                                   verbose=30, multinom=False,
                                   maxtime=20)
    print('Input parameters: {0}'.format(input_params))
    print('Optimized parameters: {0}'.format(popt))
    
    # Plot inferred DFE. Note that this will render slowly, because grid of
    # gammas is fairly dense.
    fig = plt.figure(231, figsize=(4,3), dpi=150)
    fig.clear()
    ax = fig.add_subplot(1,1,1)
    Plotting.plot_biv_dfe(s2.gammas, s2.gammas, sel_dist, popt, ax=ax)
    fig.tight_layout()
    
    #
    # Test point mass of positive selection. To do so, we test against
    # the single-population case using very high correlation.
    #
    params = [-0.5,0.5,0.99, 0.1, 4.3]
    fs_biv = s2.integrate_symmetric_point_pos(params, None, PDFs.biv_lognormal, theta,
                                             pts=None)
    
    func_single_ex = DemogSelModels.IM_single_gamma
    try:
        s1 = pickle.load(open('test.spectra1d.bpkl', 'rb'))
    except IOError:
        s1 = Cache1D(demo_params, ns, func_single_ex, pts_l=pts_l,
                     gamma_pts=100, gamma_bounds=(1e-2, 10), cpus=4,
                     additional_gammas = [1.2, 4.3], verbose=False)
        fid = open('test.spectra1d.bpkl', 'wb')
        pickle.dump(s1, fid, protocol=2)
        fid.close()
    
    fs1 = s1.integrate_point_pos([-0.5,0.5,0.1,4.3], None, PDFs.lognormal,
                                 1e5, func_single_ex)
    
    fig = dadi.Plotting.pylab.figure(229)
    fig.clear()
    dadi.Plotting.plot_2d_comp_Poisson(fs1, fs_biv, show=False)
    
    #
    # Test optimization of point mass positive selection.
    #
    
    # Generate test data set to fit
    # This is a symmetric case, with mu1=mu2=0.5, sigma1=sigma2=0.3, rho=-0.5,
    # ppos1=ppos2=0.2, gammapos1=gammapos2=1.2.
    input_params, theta = [0.5,0.3,-0.5,0.2,1.2], 1e5
    # Expected sfs
    target = s2.integrate_symmetric_point_pos(input_params, None, sel_dist, theta,
                                             pts=None)
    # Get data with Poisson variance around expectation
    data = target.sample()
    
    # We'll fit using our special-case symmetric function. The last
    # two arguments are ppos and gammapos. The first three are thus for the
    # lognormal distribution. Note that our lognormal distribution assumes
    # symmetry if the length of the arguments is only three. If we wanted
    # asymmetric lognormal, we would pass in a p0 of total length 7.
    p0 = [0.3,0.3,0.1,0.2,1.2]
    popt,llopt = dadi.Inference.opt(p0, data,
                                   s2.integrate_symmetric_point_pos,
                                   pts=None, func_args=[sel_dist, theta],
                                   # Note that mu in principle has no lower or
                                   # upper bound, sigma has only a lower bound
                                   # of 0, ppos is bounded between 0 and 1, and
                                   # gamma pos is bounded from below by 0.
                                   lower_bound=[-1,0.1,-1,0,0],
                                   upper_bound=[1,1,1,1,None],
                                   # We fix the gammapos to be 1.2, because
                                   # we can't do this integration effectively
                                   # if gammapos is allowed to vary.
                                   fixed_params=[None,None,None,None,1.2],
                                   verbose=30, multinom=False,
                                   maxtime=30)
    print('Symmetric test fit')
    print('  Input parameters: {0}'.format(input_params))
    print('  Optimized parameters: {0}'.format(popt))
    
    #
    # Mixture model
    #
    
    # Now a mixture model, which adds together a 2D distribution and a 
    # perfectly correlated 1D distribution.
    # Input parameters here a mu, sigma, rho for 2D (fixed to zero), 
    #   proportion positive, gamma positive, proportion 2D, 
    input_params, theta = [0.5,0.3,0,0.2,1.2,0.2], 1e5
    # Expected sfs
    target = mixture_symmetric_point_pos(input_params,None,s1,s2,PDFs.lognormal,
                                         PDFs.biv_lognormal, theta)
    p0 = [0.3,0.3,0,0.2,1.2,0.3]
    popt,llopt = dadi.Inference.opt(p0, data, mixture_symmetric_point_pos, pts=None, 
            func_args=[s1, s2, PDFs.lognormal,
                PDFs.biv_lognormal, theta],
            lower_bound=[None, 0.1,-1,0,None, 0],
            upper_bound=[None,None, 1,1,None, 1],
            # We fix both the rho assumed for the 2D distribution,
            # and the assumed value of positive selection.
            fixed_params=[None,None,0,None,1.2,None],
            verbose=30, multinom=False, maxtime=60)
    
    #
    # Test Godambe code for estimating uncertainties
    #
    input_params = [0.3,0.3,0.1,0.2,1.2]
    # Generate data in segments for future bootstrapping
    fs0 = s2.integrate_symmetric_point_pos(input_params, None, sel_dist,
                                          theta/100., pts=None)
    # The multiplication of fs0 is to create a range of data size among
    # bootstrap chunks, which creates a range of thetas in the bootstrap
    # data sets.
    data_pieces = [(fs0*(0.5 + (1.5-0.5)/99*ii)).sample() for ii in range(100)]
    # Add up those segments to get our data spectrum
    data = dadi.Spectrum(np.sum(data_pieces, axis=0))
    # Do the optimization
    popt,llopt = dadi.Inference.opt([0.2,0.2,0.15,0.3,1.2], data,
                                   s2.integrate_symmetric_point_pos,
                                   pts=None, func_args=[sel_dist, theta],
                                   lower_bound=[-1,0.1,-1,0,0],
                                   upper_bound=[1,1,1,1,None],
                                   fixed_params=[None,None,None,None,1.2],
                                   verbose=30, multinom=False)
    
    print('Symmetric test fit')
    print('  Input parameters: {0}'.format(input_params))
    print('  Optimized parameters: {0}'.format(popt))
    
    # Generate bootstraps for Godambe
    all_boot = []
    for boot_ii in range(100):
        # Each bootstrap is made by sampling, with replacement, from our data
        # pieces
        this_pieces = [random.choice(data_pieces) for _ in range(100)]
        all_boot.append(dadi.Spectrum(np.sum(this_pieces, axis=0)))
    
    # The Godambe methods only accept a basic dadi function that takes in
    # parameters, ns, and pts. Moreover, we can't allow gammapos to vary.
    # (We don't have variations around those cached. We could work around this
    # by pre-caching the necessary values, but it would be inelegant.) So
    # we create this simple function that holds gammapos constant and removes
    # it from the argument list.
    def temp_func(pin, ns, pts):
        # Add in gammapos parameter
        params = np.concatenate([pin, [1.2]])
        return s2.integrate_symmetric_point_pos(params, None, sel_dist,
                                               theta, pts=None)
    
    # Run the uncertainty analysis. Note that each bootstrap data set
    # needs a different assumed theta. We estimate theta for each bootstrap
    # data set simply by scaling theta from the orignal data.
    import dadi.Godambe
    boot_theta_adjusts = [b.sum()/data.sum() for b in all_boot]
    uncerts_adj = dadi.Godambe.GIM_uncert(temp_func, [], all_boot, popt[:-1],
                                          data, multinom=False,
                                          boot_theta_adjusts=boot_theta_adjusts)
    print('Godambe uncertainty test')
    print('  Input parameters: {0}'.format(input_params))
    print('  Optimized parameters: {0}'.format(popt))
    print('  Estimated 95% uncerts (theta adj): {0}'.format(1.96*uncerts_adj))
    
    # Ensure plots show up on screen.
    plt.show()
    
