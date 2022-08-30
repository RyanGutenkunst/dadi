"""
Initially developed by Bernard Kim and published as fitdadi
Kim, Huber, Lohmueller (2017) Genetics.
Modified version of the script found at: https://doi.org/10.1534/genetics.116.197145 .
Based on scripts from:
https://groups.google.com/forum/#!topic/dadi-user/4xspqlITcvc .
"""

import operator
import sys, traceback
import numpy as np
import scipy.stats.distributions
import scipy.integrate
import dadi
from dadi import Numerics, Spectrum

class Cache1D:
    def __init__(self, params, ns, demo_sel_func, pts, 
                 gamma_bounds=(1e-4, 2000), gamma_pts=500, 
                 additional_gammas=[],
                 cpus=None, gpus=0, verbose=False):
        """
        params: Optimized demographic parameters
        ns: Sample size(s) for cached spectra
        demo_sel_func: DaDi demographic function with selection. 
                       gamma must be the last argument.
        pts: Integration/extrapolation grid points settings for demo_sel_func
        gamma_bounds: Range of gammas to cache spectra for.
        gamma_pts: Number of gamma grid points over which to integrate
        additional_gammas: Additional positive values of gamma to cache for
        cpus: For multiprocessing, number of CPU jobs to launch.
              If None (default), then all CPUs available will be used.
        gpus: For multiprocessing, number of GPU jobs to launch.
        verbose: If True, print messages to track progress of cache generation.
        """
        self.params, self.ns, self.pts = tuple(params), tuple(ns), tuple(pts)

        #Create a vector of gammas that are log-spaced over an interval
        self.gammas = -np.logspace(np.log10(gamma_bounds[1]),
                                   np.log10(gamma_bounds[0]), gamma_pts)

        # Record negative gammas, for later use
        self.neg_gammas = self.gammas
        # Add additional gammas to cache
        self.gammas = np.concatenate((self.gammas, additional_gammas))
        self.spectra = [None]*len(self.gammas)
        self.params = params
        self.ns = ns
        self.pts = pts
        self.func_name = demo_sel_func.__name__

        if cpus is None:
            import multiprocessing
            cpus = multiprocessing.cpu_count()

        if not cpus > 1 or gpus > 0: #for running with a single thread
            self._single_process(verbose, demo_sel_func)
        else: #for running with with multiple cores
            self._multiple_processes(cpus, gpus, verbose, demo_sel_func)

        demo_sel_extrap_func = Numerics.make_extrap_func(demo_sel_func)
        self.neu_spec = demo_sel_extrap_func(tuple(self.params)+(0,), self.ns, self.pts)
        self.spectra = np.array(self.spectra)

    def _single_process(self, verbose, demo_sel_func):
        demo_sel_extrap_func = Numerics.make_extrap_func(demo_sel_func)
        for ii, gamma in enumerate(self.gammas):
            self.spectra[ii] = demo_sel_extrap_func(tuple(self.params)+(gamma,), self.ns,
                                             self.pts)
            if verbose:
               print('{0}: {1}'.format(ii, gamma))

    def _multiple_processes(self, cpus, gpus, verbose, demo_sel_func):
        from multiprocessing import Manager, Process, cpu_count

        with Manager() as manager:
            work = manager.Queue(cpus + gpus)
            results = manager.list()

            # Assemble pool of workers
            pool = []
            for ii in range(cpus):
                p = Process(target=self._worker_sfs,
                            args=(work, results, demo_sel_func, self.params, self.ns, self.pts, verbose, False))
                p.start()
                pool.append(p)
            for ii in range(gpus):
                p = Process(target=self._worker_sfs,
                            args=(work, results, demo_sel_func, self.params, self.ns, self.pts, verbose, True))
                p.start()
                pool.append(p)

            # Put all jobs on queue
            for ii, gamma in enumerate(self.gammas):
                work.put((ii, gamma))
            # Put commands on queue to close out workers
            for jj in range(cpus+gpus):
                work.put(None)
            # Stop workers
            for p in pool:
                p.join()    
            # Collect results
            for ii, sfs in results:
                self.spectra[ii] = sfs

    def _worker_sfs(self, in_queue, outlist, popn_func, params, ns, pts, verbose, usegpu):
        """
        Worker function -- used to generate SFSes for
        single values of gamma.
        """
        popn_func_ex = Numerics.make_extrap_func(popn_func)
        dadi.cuda_enabled(usegpu)
        while True:
            item = in_queue.get()
            if item is None:
                return
            ii, gamma = item
            try:
                sfs = popn_func_ex(tuple(params)+(gamma,), ns, pts)
                if verbose:
                    print('{0}: {1}'.format(ii, gamma))
                outlist.append((ii, sfs))
            except BaseException as inst:
                # If an exception occurs in the worker function, print an error
                # and populate the outlist with an item that will cause a later crash.
                tb = sys.exc_info()[2]
                traceback.print_tb(tb)
                outlist.append(inst)

    def integrate(self, params, ns, sel_dist, theta, pts=None, exterior_int=True):
        """
        Integrate spectra over a univariate prob. dist. for negative gammas.

        params: Parameters for sel_dist
        ns: Ignored
        sel_dist: Univariate probability distribution,
                  taking in arguments (xx, params)
        theta: Population-scaled mutation rate
        pts: Ignored
        exterior_int: If False, do not integrate outside sampled domain.

        Note also that the ns and pts arguments are ignored. They are only
        present for compatibility with other dadi functions that apply to
        demographic models.
        """
        # Restrict ourselves to negative gammas
        Nneg = len(self.neg_gammas)
        spectra = self.spectra[:Nneg]

        # Weights for integration
        weights = sel_dist(-self.neg_gammas, params)

        weighted_spectra = 0*spectra
        for ii, w in enumerate(weights):
            weighted_spectra[ii] = w*spectra[ii]

        fs = np.trapz(weighted_spectra, self.neg_gammas, axis=0)
        if not exterior_int:
            return Spectrum(theta*fs)

        smallest_gamma = self.neg_gammas[-1]
        largest_gamma = self.neg_gammas[0]
        # Compute weight for the effectively neutral portion. Not using
        # CDF function because want this to be able to compute weight
        # for arbitrary mass functions
        weight_neu, err_neu = scipy.integrate.quad(sel_dist, 0, -smallest_gamma,
                                                   args=params)
        # compute weight for the effectively lethal portion
        weight_del, err = scipy.integrate.quad(sel_dist, -largest_gamma, np.inf,
                                               args=params)

        fs += self.neu_spec*weight_neu
        fs += spectra[0]*weight_del

        return Spectrum(theta*fs)

    def integrate_point_pos(self, params, ns, sel_dist, theta, demo_sel_func=None, 
                            Npos=1, pts=None, exterior_int=True):
        """
        Integrate spectra over a univariate prob. dist. for negative gammas,
        plus one or more point masses of positive selection.

        params: Parameters. The last Npos*2 are assumed to be the proportion of
                positive selection and the gamma for each point mass, in the order
                (ppos1, gammapos1, ppos2, gammapos2, ...).
                The remaining parameters are for the continuous point mass.
        ns: Ignored
        sel_dist: Univariate probability distribution,
                  taking in arguments (xx, params)
        theta: Population-scaled mutation rate
        demo_sel_func: DaDi demographic function with selection. 
                       gamma must be the last argument.
        Npos: Number of positive point masses to model.
        pts: Ignored, evaluation of demo_self_func will use pts from orignal
               caching.
        exterior_int: If False, do not integrate outside sampled domain.

        Note also that the ns and pts arguments are ignored. They are only
        present for compatibility with other dadi functions that apply to
        demographic models.
        """
        pdf_params, ppos, gammapos = params[:-2*Npos], params[-2], params[-1]
        ppos_l, gammapos_l = params[-2*Npos::2], params[-2*Npos+1::2]

        pdf_fs = self.integrate(pdf_params, None, sel_dist, theta, None,
                                exterior_int=exterior_int)
        result = (1-np.sum(ppos_l))*pdf_fs
        
        if demo_sel_func is not None:
            demo_sel_func = Numerics.make_extrap_func(demo_sel_func)

        for ppos, gammapos in zip(ppos_l, gammapos_l):
            if gammapos not in self.gammas:
                if demo_sel_func is None:
                    raise IndexError('Failed to find requested gammapos={0:.4f} '
                                     'in Cache1D spectra. Was it included in '
                                     'additional_gammas during cache generation?'.format(gammapos))
                pos_fs = theta*demo_sel_func(tuple(self.params) + (gammapos,),
                                             self.ns, self.pts)
                self.gammas = np.append(self.gammas, gammapos)
                self.spectra = np.append(self.spectra, [pos_fs.data], axis=0)
            ii = list(self.gammas).index(gammapos)
            pos_fs = Spectrum(self.spectra[ii])
            result += ppos*pos_fs

        return result
