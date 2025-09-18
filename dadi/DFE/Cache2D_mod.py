"""
Developed by the Gutenkunst group, building off of the fitdadi code.
"""
import sys, traceback
import numpy as np
try:
    np.trapz = np.trapezoid
except AttributeError:
    pass
import scipy.integrate

class Cache2D:
    def __init__(self, params, ns, demo_sel_func, pts,
                 gamma_bounds=(1e-4, 2000.), gamma_pts=100,
                 additional_gammas=[], cpus=None, gpus=0, verbose=False,
                 split_jobs=1, this_job_id=0):
        """
        Initialize the Cache2D object.

        Args:
            params (list): Optimized demographic parameters.

            ns (list): Sample sizes for cached spectra.

            demo_sel_func (function): DaDi demographic function with selection.
                gamma1, gamma2 must be the last arguments.

            pts (list): Grid point settings for demo_sel_func.

            gamma_bounds (tuple, optional): Range of gammas to integrate over.
                Defaults to (1e-4, 2000.).

            gamma_pts (int, optional): Number of gamma grid points over which
                to integrate. Defaults to 100.

            additional_gammas (list, optional): Sequence of additional gamma
                values to store results for. Useful for point masses of explicit
                neutrality or positive selection. Defaults to [].

            cpus (int, optional): Number of CPU jobs to launch. If None, all
                available CPUs will be used. Defaults to None.

            gpus (int, optional): Number of GPU jobs to launch. Defaults to 0.

            verbose (bool, optional): If True, print messages to track progress
                of cache generation. Defaults to False.

            split_jobs (int, optional): To split cache generation across
                multiple computing jobs, set split_jobs > 1. Defaults to 1.

            this_job_id (int, optional): Defines which entry in split_jobs this
                run will create (indexed from 0). Defaults to 0.

        Example:
            To split Cache2D generation over 3 independent jobs, set
            split_jobs=3 and create jobs with this_job_id=0, 1, 2.
            Then use Cache2D.merge to combine the outputs.
        """
        # Store details regarding how this cache was generated. May be useful
        # for keeping track of pickled caches.
        self.ns = ns
        self.pts = pts
        self.func_name = demo_sel_func.__name__
        self.params = params

        # Create a vector of gammas that are log-spaced over sequential 
        # intervals or log-spaced over a single interval.
        self.gammas = -np.logspace(np.log10(gamma_bounds[1]),
                                   np.log10(gamma_bounds[0]), gamma_pts)

        # Store bulk negative gammas for use in integration of negative pdf
        self.neg_gammas = self.gammas
        # Add additional gammas to array
        self.gammas = np.concatenate((self.gammas, additional_gammas))

        self.spectra = [[None]*len(self.gammas) for _ in self.gammas]

        if cpus is None:
            import multiprocessing
            cpus = multiprocessing.cpu_count()

        if not cpus > 1 or gpus > 0: #for running with a single thread
            self._single_process(verbose, split_jobs, this_job_id, demo_sel_func)
        else: #for running with with multiple cores
            self._multiple_processes(cpus, gpus, verbose, split_jobs, this_job_id, demo_sel_func)

        # self.spectra is an array of arrays. The first two dimensions are
        # indexed by the pairs of gamma values, and the remaining dimensions
        # are the spectra themselves.
        if split_jobs == 1:
            self.spectra = np.array(self.spectra)
        
    def _single_process(self, verbose, split_jobs, this_job_id, demo_sel_func):
        """
        Run cache generation using a single process.

        Args:
            verbose (bool): If True, print progress messages.

            split_jobs (int): Number of jobs to split the cache generation.

            this_job_id (int): ID of the current job.

            demo_sel_func (function): DaDi demographic function with selection.
        """
        func_ex = Numerics.make_extrap_func(demo_sel_func)

        this_eval = 0
        for ii,gamma in enumerate(self.gammas):
            for jj, gamma2 in enumerate(self.gammas):
                if this_eval % split_jobs == this_job_id:
                    self.spectra[ii][jj] = func_ex(tuple(self.params)+(gamma,gamma2),
                            self.ns, self.pts)
                    if verbose: print('{0},{1}: {2},{3}'.format(ii,jj, gamma,gamma2))
                this_eval += 1
                        
    def _multiple_processes(self, cpus, gpus, verbose, split_jobs, this_job_id, demo_sel_func):
        """
        Run cache generation using multiple processes.

        Args:
            cpus (int): Number of CPU jobs to launch.

            gpus (int): Number of GPU jobs to launch.

            verbose (bool): If True, print progress messages.

            split_jobs (int): Number of jobs to split the cache generation.

            this_job_id (int): ID of the current job.

            demo_sel_func (function): DaDi demographic function with selection.
        """
        from multiprocessing import Manager, Process

        with Manager() as manager:
            work = manager.Queue(cpus+gpus)
            results = manager.list()
                
            # Assemble pool of workers
            pool = []
            for i in range(cpus):
                p = Process(target=self._worker_sfs,
                            args=(work, results, demo_sel_func, self.params, self.ns, self.pts, verbose, False))
                p.start()
                pool.append(p)
            for i in range(gpus):
                p = Process(target=self._worker_sfs,
                            args=(work, results, demo_sel_func, self.params, self.ns, self.pts, verbose, True))
                p.start()
                pool.append(p)

            # Put all jobs on queue
            this_eval = 0
            for ii, gamma in enumerate(self.gammas):
                for jj, gamma2 in enumerate(self.gammas):
                    if this_eval % split_jobs == this_job_id:
                        work.put((ii,jj, gamma,gamma2))
                    this_eval += 1
            # Put commands on queue to close out workers
            for jj in range(cpus+gpus):
                work.put(None)
            # Stop workers
            for p in pool:
                p.join()
            for ii,jj, sfs in results:
                self.spectra[ii][jj] = sfs
                    
    def _worker_sfs(self, in_queue, outlist, demo_sel_func, params, ns, pts, verbose, usegpu):
        """
        Worker function to generate SFSes for pairs of gammas.

        Args:
            in_queue (Queue): Input queue for tasks.

            outlist (list): List to store results.

            demo_sel_func (function): DaDi demographic function with selection.

            params (list): Parameters for the demographic function.

            ns (list): Sample sizes for cached spectra.

            pts (list): Grid point settings for demo_sel_func.

            verbose (bool): If True, print progress messages.

            usegpu (bool): If True, use GPU for computation.
        """
        demo_sel_func = Numerics.make_extrap_func(demo_sel_func)
        dadi.cuda_enabled(usegpu)
        while True:
            item = in_queue.get()
            if item is None:
                return
            ii, jj, gamma, gamma2 = item
            try:
                sfs = demo_sel_func(tuple(params)+(gamma,gamma2), ns, pts)
                if verbose:
                    print('{0},{1}: {2},{3}'.format(ii, jj, gamma, gamma2))
                outlist.append((ii,jj,sfs))
            except BaseException as inst:
                # If an exception occurs in the worker function, print an error
                # and populate the outlist with an item that will cause a later crash.
                tb = sys.exc_info()[2]
                traceback.print_tb(tb)
                outlist.append(inst)

    def integrate(self, params, ns, sel_dist, theta, pts,
                  exterior_int=True):
        """
        Integrate spectra over a bivariate probability distribution for negative gammas.

        Args:
            params (list): Parameters for sel_dist.

            ns (list): Ignored.

            sel_dist (function): Bivariate probability distribution, taking
                arguments (xx, yy, params).

            theta (float): Population-scaled mutation rate.

            pts (list): Ignored.

            exterior_int (bool, optional): If False, do not integrate outside
                the sampled domain. Defaults to True.

        Returns:
            fs (Spectrum): Integrated spectrum.
        """
        # Restrict our gammas and spectra to negative gammas.
        Nneg = len(self.neg_gammas)
        spectra = self.spectra[:Nneg, :Nneg]
        params = np.array(params)

        # Weights for integration
        weights = sel_dist(-self.neg_gammas, -self.neg_gammas, params)
        # Apply weighting. Could probably do this in a single fancy numpy
        # multiplication, which might be faster.
        weighted_spectra = 0*spectra
        for ii, row in enumerate(weights):
            for jj, w in enumerate(row):
                weighted_spectra[ii,jj] = w*spectra[ii,jj]

        # Integrate out the first two axes, which are the gamma axes.
        temp = np.trapz(weighted_spectra, self.neg_gammas, axis=0)
        fs = np.trapz(temp, self.neg_gammas, axis=0)

        if not exterior_int:
            return Spectrum(theta*fs)

        # Test whether our DFE pdf is symmetric. If it is we can dramatically reduce our calculations.
        testx = np.logspace(-2,2,3)
        testout = sel_dist(testx, testx, params)
        # We need atol=0 here to ensure small values don't lead to spurious pass of test
        symmetric_dfe = np.allclose(testout, testout.T, atol=0, rtol=1e-12)

        max_gamma = -self.neg_gammas[-1]
        min_gamma = -self.neg_gammas[0]
        # Account for density that is outside the simulated domain.
        weights_1low, weights_1high = 0*self.neg_gammas, 0*self.neg_gammas
        weights_2low, weights_2high = 0*self.neg_gammas, 0*self.neg_gammas
        for ii, gamma in enumerate(-self.neg_gammas):
            w_1low, err = scipy.integrate.quad(sel_dist, min_gamma, np.inf, epsabs=1e-4, epsrel=1e-3, args=(gamma,params))
            w_1high, err = scipy.integrate.quad(sel_dist, 0, max_gamma, epsabs=1e-4, epsrel=1e-3, args=(gamma,params))
            if symmetric_dfe:
                w_2low, w_2high = w_1low, w_1high
            else:
                marg2_func = lambda g2: sel_dist(gamma, g2, params)
                w_2low, err = scipy.integrate.quad(marg2_func, min_gamma, np.inf, epsabs=1e-4, epsrel=1e-3)
                w_2high, err = scipy.integrate.quad(marg2_func, 0, max_gamma, epsabs=1e-4, epsrel=1e-3)
            weights_1low[ii] = w_1low
            weights_2low[ii] = w_2low
            weights_1high[ii] = w_1high
            weights_2high[ii] = w_2high

        # Strongly deleterious gamma2, in-range gamma1
        fs += np.trapz(spectra[:,0] * weights_2low[:,np.newaxis,np.newaxis],
                       self.neg_gammas, axis=0)
        # Neutral gamma2, in-range gamma1
        fs += np.trapz(spectra[:,-1] * weights_2high[:,np.newaxis,np.newaxis],
                       self.neg_gammas, axis=0)
        # Strongly deleterious gamma1, in-range gamma2
        fs += np.trapz(spectra[0,:] * weights_1low[:,np.newaxis,np.newaxis],
                       self.neg_gammas, axis=0)
        # Neutral gamma1, in-range gamma2
        fs += np.trapz(spectra[-1,:] * weights_1high[:,np.newaxis,np.newaxis],
                       self.neg_gammas, axis=0)

        # Both neutral, really slow
        weight, err = scipy.integrate.dblquad(sel_dist, 0, max_gamma,
                                              lambda _: 0, lambda _: max_gamma,
                                              epsrel=1e-3, epsabs=1e-4, args=[params])
        fs += spectra[-1,-1]*weight

        # Neutral gamma2, strongly deleterious gamma1
        weight, err = scipy.integrate.dblquad(sel_dist, 0, max_gamma,
                lambda _: min_gamma, lambda _: np.inf,
                epsrel=1e-3, epsabs=1e-4 ,args=[params])
        fs += spectra[0,-1]*weight

        # Neutral gamma1, strongly deleterious gamma2
        if not symmetric_dfe:
            weight, err = scipy.integrate.dblquad(sel_dist, min_gamma, np.inf,
                    lambda _: 0, lambda _: max_gamma,
                    epsrel=1e-3, epsabs=1e-4, args=[params])
        fs += spectra[-1,0]*weight

        return Spectrum(theta*fs)

    def integrate_point_pos(self, params, ns, biv_seldist, theta,
                            rho=0, pts=None):
        """
        Integrate spectra over a bivariate prob. dist. for negative gammas plus
        a point mass of positive selection.

        Note that no normalization is performed, so alleles not covered by the
        specified range of gammas are assumed not to be seen in the data.

        Note that in the triallelic paper (Ragsdale et al. 2016 Genetics), we
        weighted each portion of the DFE based on rho. This was to ensure that
        the rho = 0 limit corresponded to independent sampling and the rho = 1
        limit corresponding to exactly equal selection coefficients for each
        pair of derived alleles.

        If we generalize to allow the marginal DFEs to differ between the
        two populations, the rho = 1 limit cannot be held perfectly between
        both negative and positive selection quadrants. (In log-space, the
        positive selection mass is infinitely far from the negative selection
        DFE.)

        We do use a similar procedure to the triallelic paper to end up
        with something like:
        
        - p++ = p1 * p2 + rho * (sqrt(p1 * p2) - p1 * p2)
        
        - p+- = (1 - rho) * p1 * (1 - p2)
        
        - p-- = (1 - p1) * (1 - p2) + rho * (1 - sqrt(p1 * p2) - (1 - p1) * (1 - p2))

        The logic here is that in the rho=1 limit, we set the proportion
        positively selected to be the geometric mean of p1 and p2, the
        proportion positive in one pop and negative in the other (p+-) to be
        zero, and the remainder negative in both populations p--. We then
        linearly interpolate between the rho=0 and rho=1 cases.

        This requires the integration method to explicitly know about rho,
        so it's not completely general to all joint DFEs. rho is thus
        included as a parameter in the argument list.

        Args:
            params (list): Parameters for sel_dist and positive selection.
                The last four parameters are:
                
                - Proportion positive selection in pop1.
                
                - Positive gamma for pop1.
                
                - Proportion positive in pop2.
                
                - Positive gamma for pop2.

            ns (list): Ignored.

            biv_seldist (function): Bivariate probability distribution for
                negative selection, taking arguments (xx, yy, params).

            theta (float): Population-scaled mutation rate.

            rho (float, optional): Correlation coefficient used to connect
                negative and positive components of DFE. Defaults to 0.

            pts (list, optional): Ignored.

        Returns:
            fs (Spectrum): Integrated spectrum.
        """
        biv_params = params[:-4]
        ppos1, gammapos1, ppos2, gammapos2 = params[-4:]

        Nneg = len(self.neg_gammas)
        weights = biv_seldist(-self.neg_gammas, -self.neg_gammas, biv_params)

        # Case in which both are negative, so we integrate directly
        # over the bivariate distribution
        neg_neg = self.integrate(biv_params, ns, biv_seldist, 1, pts)

        # Case in which both are positive, which is a single spectrum.
        try:
            pos_pos = self.spectra[self.gammas == gammapos1,
                                   self.gammas == gammapos2][0]
        except IndexError:
            raise IndexError('Failed to find requested gammapos1={0:.4f} '
                             'and/or gammapos2={1:0.4f} in cached spectra. '
                             'Were they included in additional_gammas during '
                             'cache generation?'.format(gammapos1,gammapos2))

        # Case in which pop1 is positive and pop2 is negative.
        pos_neg_spectra = np.squeeze(self.spectra[self.gammas==gammapos1,:Nneg])
        # Obtain the marginal DFE for pop2 by integrating over the weights
        # for pop1.
        pos_neg_weights = np.trapz(weights, self.neg_gammas, axis=0)
        # For numpy multiplication...
        pos_neg_weights = pos_neg_weights[:,np.newaxis,np.newaxis]
        # Do the weighted integral.
        pos_neg = np.trapz(pos_neg_weights*pos_neg_spectra,
                           self.neg_gammas, axis=0)

        # Case in which pop2 is positive and pop1 is negative.
        neg_pos_spectra = np.squeeze(self.spectra[:Nneg,self.gammas==gammapos2])
        # Now integrate over pop2 to get marginal DFE for pop1
        neg_pos_weights = np.trapz(weights, self.neg_gammas, axis=1)
        neg_pos_weights = neg_pos_weights[:,np.newaxis,np.newaxis]
        neg_pos = np.trapz(neg_pos_weights*neg_pos_spectra,
                           self.neg_gammas, axis=0)

        # Weighting factors for each quadrant of the DFE. Note that in the case
        # that ppos1==ppos2, these reduce to the model of Ragsdale et al. (2016)
        p_pos_pos = ppos1*ppos2 + rho*(np.sqrt(ppos1*ppos2) - ppos1*ppos2)
        p_pos_neg = (1-rho) * ppos1*(1-ppos2)
        p_neg_pos = (1-rho) * (1-ppos1)*ppos2
        p_neg_neg = (1-ppos1)*(1-ppos2)\
                + rho*(1-np.sqrt(ppos1*ppos2) - (1-ppos1)*(1-ppos2))

        fs = p_pos_pos*pos_pos + p_pos_neg*pos_neg + p_neg_pos*neg_pos\
                + p_neg_neg*neg_neg
        return theta*fs

    def integrate_symmetric_point_pos(self, params, ns, biv_seldist, theta, pts=None):
        """
        Integrate spectra over a bivariate probability distribution for negative gammas
        plus a symmetric point mass of positive selection.

        Args:
            params (list): Parameters for sel_dist and positive selection.
                The last two parameters are:
                
                - Proportion positive selection.
                
                - Positive gamma.
                
                Earlier arguments are assumed to be for the continuous bivariate
                distribution. The last of those earlier arguments is the
                correlation coefficient rho.

            ns (list): Ignored.

            biv_seldist (function): Bivariate probability distribution for
                negative selection, taking arguments (xx, yy, params).

            theta (float): Population-scaled mutation rate.

            pts (list, optional): Ignored.

        Returns:
            fs (Spectrum): Integrated spectrum.
        """
        seldist_params = params[:-2]
        rho = seldist_params[-1]
        ppos, gammapos = params[-2:]

        params = np.concatenate((seldist_params, [ppos,gammapos,ppos,gammapos]))

        return self.integrate_point_pos(params, ns, biv_seldist, theta,
                                        rho=rho, pts=None)

    @staticmethod
    def merge(caches):
        """
        Merge caches generated with split_jobs.

        Args:
            caches (list): List of Cache2D objects to merge.

        Returns:
            cache (Cache2D): Merged Cache2D object.

        Raises:
            ValueError: If caches conflict or do not merge into a complete cache.
        """
        import copy
        # Copy our first cache to start the output
        new_cache = copy.deepcopy(caches[0])
        for other in caches[1:]:
            for ii, row in enumerate(other.spectra):
                for jj, fs in enumerate(row):
                    if fs is not None:
                        if new_cache.spectra[ii][jj] is not None \
                                and not np.all(new_cache.spectra[ii][jj] == fs):
                            raise ValueError("Merged cached conflicts with current.")
                        new_cache.spectra[ii][jj] = fs
        # Check that new cache is complete
        for ii, row in enumerate(new_cache.spectra):
            for jj, fs in enumerate(row):
                if fs is None:
                    raise ValueError("Cache is incomplete after merging. "
                            "First missing entry is {0},{1}.".format(ii,jj))
        # Convert cache spectra to array
        new_cache.spectra = np.array(new_cache.spectra)
        return new_cache
#
# Example demography + selection functions
#
import dadi
from dadi import Numerics, PhiManip, Integration
from dadi.Spectrum_mod import Spectrum

def mixture(params, ns, s1, s2, sel_dist1, sel_dist2, theta, pts,
            exterior_int=True):
    """
    Compute a weighted summation of 1D and 2D distributions that share parameters.

    The 1D distribution assumes selection coefficients are perfectly correlated.

    Args:
        params (list): Parameters for optimization. The last parameter is the 
            weight for the 2D distribution. The second-to-last parameter is the 
            correlation coefficient for the 2D distribution. The remaining 
            parameters are shared between the 1D and 2D distributions.

        ns (list): Ignored. Will be retrieved from original caching.

        s1 (Cache1D): Cache object for the 1D distribution.

        s2 (Cache2D): Cache object for the 2D distribution.

        sel_dist1 (function): Univariate probability distribution for `s1`.

        sel_dist2 (function): Bivariate probability distribution for `s2`.

        theta (float): Population-scaled mutation rate.

        pts (list): Ignored. Will be retrieved from original caching.

        exterior_int (bool, optional): If False, do not integrate outside the 
            sampled domain. Defaults to True.

    Returns:
        fs (Spectrum): Weighted summation of the 1D and 2D distributions.
    """
    fs1 = s1.integrate(params[:-2], None, sel_dist1, theta, None, exterior_int)
    fs2 = s2.integrate(params[:-1], None, sel_dist2, theta, None, exterior_int)

    p2d = params[-1]
    return (1-p2d)*fs1 + p2d*fs2

def mixture_symmetric_point_pos(params, ns, s1, s2, sel_dist1, sel_dist2,
                                theta, pts=None):
    """
    Compute a weighted summation of 1D and 2D distributions with positive selection.

    The 1D distribution assumes selection coefficients are perfectly correlated.

    Args:
        params (list): Parameters for optimization. The last parameter is the 
            weight for the 2D distribution. The second-to-last parameter is the 
            positive gamma for the point mass. The third-to-last parameter is 
            the proportion of positive selection. The fourth-to-last parameter 
            is the correlation coefficient for the 2D distribution. The remaining 
            parameters are shared between the 1D and 2D distributions.

        ns (list): Ignored. Will be retrieved from original caching.

        s1 (Cache1D): Cache object for the 1D distribution.

        s2 (Cache2D): Cache object for the 2D distribution.

        sel_dist1 (function): Univariate probability distribution for `s1`.

        sel_dist2 (function): Bivariate probability distribution for `s2`.

        theta (float): Population-scaled mutation rate.

        pts (list, optional): Ignored. Will be retrieved from original caching.

    Returns:
        fs (Spectrum): Weighted summation of the 1D and 2D distributions with positive selection.
    """
    pdf_params = params[:-4]
    rho, ppos, gamma_pos, p2d = params[-4:]

    params1 = list(pdf_params) + [ppos, gamma_pos]
    fs1 = s1.integrate_point_pos(params1, None, sel_dist1, theta, Npos=1, pts=None)
    params2 = list(pdf_params) + [rho, ppos, gamma_pos, ppos, gamma_pos]
    fs2 = s2.integrate_symmetric_point_pos(params2, None, sel_dist2, theta, None)
    return (1-p2d)*fs1 + p2d*fs2

def mixture_point_pos(params, ns, s1, s2, sel_dist1, sel_dist2,
                      theta, pts=None):
    """
    Compute a weighted summation of 1D and 2D distributions with positive selection.

    The 1D distribution assumes selection coefficients are perfectly correlated.

    Args:
        params (list): Parameters for optimization. The last parameter is the 
            weight for the 2D distribution. The second-to-last parameter is the 
            positive gamma for the point mass in population 2. The third-to-last 
            parameter is the proportion of positive selection in population 2. 
            The fourth-to-last parameter is the positive gamma for the point 
            mass in population 1. The fifth-to-last parameter is the proportion 
            of positive selection in population 1. The sixth-to-last parameter 
            is the correlation coefficient for the 2D distribution. The remaining 
            parameters are shared between the 1D and 2D distributions.

        ns (list): Ignored. Will be retrieved from original caching.

        s1 (Cache1D): Cache object for the 1D distribution.

        s2 (Cache2D): Cache object for the 2D distribution.

        sel_dist1 (function): Univariate probability distribution for `s1`.

        sel_dist2 (function): Bivariate probability distribution for `s2`.

        theta (float): Population-scaled mutation rate.

        pts (list, optional): Ignored. Will be retrieved from original caching.

    Returns:
        fs (Spectrum): Weighted summation of the 1D and 2D distributions with positive selection.
    """
    pdf_params = params[:-6]
    rho, ppos1, gamma_pos1, ppos2, gamma_pos2, p2d = params[-6:]

    params1 = list(pdf_params) + [ppos1, gamma_pos1]
    fs1 = s1.integrate_point_pos(params1, None, sel_dist1, theta, Npos=1, pts=None)
    params2 = list(pdf_params) + [rho, ppos1, gamma_pos1, ppos2, gamma_pos2]
    fs2 = s2.integrate_point_pos(params2, None, sel_dist2, theta, None)
    return (1-p2d)*fs1 + p2d*fs2
