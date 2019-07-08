"""
Initially developed by Bernard Kim and published as fitdadi
Kim, Huber, Lohmueller (2017) Genetics.
Modified version of the script found at: https://doi.org/10.1534/genetics.116.197145 .
Based on scripts from:
https://groups.google.com/forum/#!topic/dadi-user/4xspqlITcvc .
"""

import operator
import numpy as np
import scipy.stats.distributions
import scipy.integrate
from dadi import Numerics, Spectrum

class Cache1D:
    def __init__(self, params, ns, demo_sel_func, pts_l, 
                 gamma_bounds=(1e-4, 2000), gamma_pts=500, 
                 additional_gammas=[],
                 mp=False, cpus=None, verbose=False):
        """
        params: Optimized demographic parameters
        ns: Sample size(s) for cached spectra
        demo_sel_func: DaDi demographic function with selection. 
                       gamma must be the last argument.
        pts_l: Integration/extrapolation grid points settings for demo_sel_func
        gamma_bounds: Range of gammas to cache spectra for.
        gamma_pts: Number of gamma grid points over which to integrate
        additional_gammas: Additional positive values of gamma to cache for
        mp: True if you want to use multiple cores (utilizes 
            multiprocessing). If using the mp option you may also specify
            # of cpus, otherwise this will just use nthreads-1 on your
            machine.
        cpus: For multiprocessing, number of jobs to launch.
        verbose: If True, print messages to track progress of cache generation.
        """
        self.params, self.ns, self.pts_l = tuple(params), tuple(ns), tuple(pts_l)

        #Create a vector of gammas that are log-spaced over an interval
        self.gammas = -np.logspace(np.log10(gamma_bounds[1]),
                                   np.log10(gamma_bounds[0]), gamma_pts)

        # Record negative gammas, for later use
        self.neg_gammas = self.gammas
        # Add additional gammas to cache
        self.gammas = np.concatenate((self.gammas, additional_gammas))
        self.spectra = [None]*len(self.gammas)

        if not mp: #for running with a single thread
            for ii, gamma in enumerate(self.gammas):
                self.spectra[ii] = demo_sel_func(tuple(params)+(gamma,), self.ns,
                                                 self.pts_l)
                if verbose:
                   print('{0}: {1}'.format(ii, gamma))
        else: #for running with with multiple cores
            import multiprocessing
            if cpus is None:
                cpus = multiprocessing.cpu_count() - 1
            def worker_sfs(in_queue, outlist, popn_func_ex, params, ns,
                           pts_l):
                """
                Worker function -- used to generate SFSes for
                single values of gamma.
                """
                while True:
                    item = in_queue.get()
                    if item is None:
                        return
                    ii, gamma = item
                    sfs = popn_func_ex(tuple(params)+(gamma,), ns, pts_l)
                    if verbose:
                        print('{0}: {1}'.format(ii, gamma))
                    outlist.append((ii, sfs))

            if __name__ ==  '__main__':        
                manager = multiprocessing.Manager()
                if cpus is None:
                    cpus = multiprocessing.cpu_count()
                work = manager.Queue(cpus-1)
                results = manager.list()

                # Assemble pool of workers
                pool = []
                for ii in range(cpus):
                    p = multiprocessing.Process(target=worker_sfs,
                                                args=(work, results, demo_sel_func, params, ns, pts_l))
                    p.start()
                    pool.append(p)

                # Put all jobs on queue
                for ii, gamma in enumerate(self.gammas):
                    work.put((ii, gamma))
                # Put commands on queue to close out workers
                for jj in range(cpus):
                    work.put(None)
                # Start work
                for p in pool:
                    p.join()
                # Collect results
                for ii, sfs in results:
                    self.spectra[ii] = sfs

        self.neu_spec = demo_sel_func(tuple(params)+(0,), ns, pts_l)
        self.spectra = np.array(self.spectra)

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
        pts: Ignored, evaluation of demo_self_func will use pts_l from orignal
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

        for ppos, gammapos in zip(ppos_l, gammapos_l):
            if gammapos not in self.gammas:
                if demo_sel_func is None:
                    raise IndexError('Failed to find requested gammapos={0:.4f} '
                                     'in Cache1D spectra. Was it included in '
                                     'additional_gammas during cache generation?'.format(gammapos))
                pos_fs = theta*demo_sel_func(tuple(self.params) + (gammapos,),
                                             self.ns, self.pts_l)
                self.gammas = np.append(self.gammas, gammapos)
                self.spectra = np.append(self.spectra, [pos_fs.data], axis=0)
            ii = list(self.gammas).index(gammapos)
            pos_fs = Spectrum(self.spectra[ii])
            result += ppos*pos_fs

        return result