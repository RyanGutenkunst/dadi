"""
Distribution functions to wrap demographics with scalar selection. Most
of this code is modified dadi code, and the selection stuff is a 
modified version of the script found at: 
https://groups.google.com/forum/#!topic/dadi-user/4xspqlITcvc

There are a few changes to the integration, so that anything below the
lower bound is assumed to be effectively neutral; and anything above
the lower bound is assumed to not be segregating, and weighted 0.

I added in multiprocessing capabilities because generating all the 
spectra takes really long for large values of gamma. One workaround
is to not integrate where the gammas are large, since the SFSes there
should be close to 0... 
"""
import numpy as np
import scipy.integrate
import scipy.special as ss

class spectra2d:
    def __init__(self, params, ns, func_ex, pts=None,
                 Npts=500, int_bounds=(1e-4, 1000.), int_breaks=None,
                 additional_gammas=[],
                 mp=False, cpus=None, echo=False):
        """
        params: Optimized demographic parameters
        ns: Sample sizes
        func_ex: dadi demographic function with selection. Gammas for the two
                 populations should be the last arguments.
        pts: Grid point settings for func_ex
        Npts: Number of gamma grid points over which to integrate
        int_bounds: Range of gammas to integrate over
        int_breaks: Break points for spacing out the gamma intervals
        additional_gammas: Sequence of additional gamma values to store results
                           for. Useful for point masses of explicit neutrality
                           or positive selection.
        mp: Set to True to use multiple cores on machine
        cpus: If mp=True, this is number of cores that will be used
        echo: If True, output after each fs evaluation, to track progress.
        """
        # Threshold to use when checking whether bivariate DFE is normalized
        # given the gamma grid used.
        self.normalization_threshold = np.inf

        # Store details regarding how this cache was generated. May be useful
        # for keeping track of pickled caches.
        self.ns = ns
        self.pts = pts
        self.func_name = func_ex.func_name
        self.params = params

        # Create a vector of gammas that are log-spaced over sequential 
        # intervals or log-spaced over a single interval.
        if int_breaks is not None:
            numbreaks = len(int_breaks)
            stepint = Npts/(numbreaks-1)
            self.gammas = []
            for i in reversed(range(0,numbreaks-1)):
                self.gammas.append(-np.logspace(np.log10(int_breaks[i+1]),
                                                np.log10(int_breaks[i]),
                                                stepint))
            self.gammas = np.concatenate(self.gammas)
        else:
            self.gammas = -np.logspace(np.log10(int_bounds[1]),
                                          np.log10(int_bounds[0]), Npts)

        # Store bulk negative gammas for use in integration of negative pdf
        self.neg_gammas = self.gammas

        # Add additional gammas to array, first removing redundant ones
        additional_gammas = set(additional_gammas).difference(self.gammas)
        additional_gammas = sorted(list(additional_gammas))
        self.gammas = np.concatenate((self.gammas, additional_gammas))

        self.params = tuple(params)

        self.spectra = [[None]*len(self.gammas) for _ in self.gammas]

        if not mp: #for running with a single thread
            for ii,gamma in enumerate(self.gammas):
                for jj, gamma2 in enumerate(self.gammas):
                    self.spectra[ii][jj] = func_ex(tuple(params)+(gamma,gamma2),
                                                   ns, pts)
                    if echo:
                        print '{0},{1}: {2},{3}'.format(ii,jj, gamma,gamma2)
        else: #for running with with multiple cores
            import multiprocessing
            # Would like to simply use Pool.map here, but difficult to
            # pass dynamic func_ex that way.

            def worker_sfs(in_queue, outlist, popn_func_ex, params, ns, pts):
                """
                Worker function -- used to generate SFSes for
                pairs of gammas.
                """
                while True:
                    item = in_queue.get()
                    if item is None:
                        return
                    ii, jj, gamma, gamma2 = item
                    sfs = popn_func_ex(tuple(params)+(gamma,gamma2), ns, pts)
                    print '{0},{1}: {2},{3}'.format(ii,jj, gamma,gamma2)
                    outlist.append((ii,jj,sfs))

            manager = multiprocessing.Manager()
            if cpus is None:
                cpus = multiprocessing.cpu_count()
            work = manager.Queue(cpus-1)
            results = manager.list()

            # Assemble pool of workers
            pool = []
            for i in xrange(cpus):
                p = multiprocessing.Process(target=worker_sfs,
                                            args=(work, results, func_ex,
                                                  params, ns, pts))
                p.start()
                pool.append(p)

            # Put all jobs on queue
            for ii, gamma in enumerate(self.gammas):
                for jj, gamma2 in enumerate(self.gammas):
                    work.put((ii,jj, gamma,gamma2))
            # Put commands on queue to close out workers
            for jj in xrange(cpus):
                work.put(None)
            # Start work
            for p in pool:
                p.join()
            for ii,jj, sfs in results:
                self.spectra[ii][jj] = sfs

        # self.spectra is an array of arrays. The first two dimensions are
        # indexed by the pairs of gamma values, and the remaining dimensions
        # are the spectra themselves.
        self.spectra = np.array(self.spectra)

    def integrate_uni(self, params, ns, sel_dist, theta, pts):
        """
        Integrate spectra over a univariate prob. dist. for negative gammas.

        params: Parameters for sel_dist
        ns: Ignored
        sel_dist: Univariate probability distribution,
                  taking in arguments (xx, params)
        theta: Population-scaled mutation rate
        pts: Ignored

        Note also that the ns and pts arguments are ignored. They are only
        present for compatibility with other dadi functions that apply to
        demographic models.
        """
        # Restrict our gammas and spectra to negative gammas.
        Nneg = len(self.neg_gammas)
        spectra = np.array([self.spectra[ii,ii] for ii in range(Nneg)])

        # Weights for integration
        # XXX: Can I make this work cleanly with 1D and 2D distributions?
        weights = np.diag(sel_dist(-self.neg_gammas, -self.neg_gammas, params))
        # Apply weighting. Could probably do this in a single fancy numpy
        # multiplication, which might be faster.
        weighted_spectra = 0*spectra
        for ii, w in enumerate(weights):
            weighted_spectra[ii] = w*spectra[ii]
        fs = np.trapz(weighted_spectra, self.neg_gammas, axis=0)

        smallest_gamma = -self.neg_gammas[-1]
        largest_gamma = -self.neg_gammas[0]
        # Account for density that is outside the simulated domain.
        weight_neu, err = scipy.integrate.quad(sel_dist, 0, smallest_gamma,
                                               args=params)
        weight_del, err = scipy.integrate.quad(sel_dist, largest_gamma, np.inf,
                                               args=params)

        fs += weight_neu*spectra[-1]
        fs += weight_del*spectra[0]

        return Spectrum(theta*fs)

    def integrate_biv(self, params, ns, sel_dist, theta, pts,
                      exterior_int=True):
        """
        Integrate spectra over a bivariate prob. dist. for negative gammas.

        params: Parameters for sel_dist
        ns: Ignored
        sel_dist: Bivariate probability distribution,
                  taking in arguments (xx, yy, params)
        theta: Population-scaled mutation rate
        pts: Ignored
        exterior_int: If False, do not integrate outside sampled domain.

        Note also that the ns and pts arguments are ignored. They are only
        present for compatibility with other dadi functions that apply to
        demographic models.
        """
        self.check_normalization(params, sel_dist)

        # Restrict our gammas and spectra to negative gammas.
        Nneg = len(self.neg_gammas)
        spectra = self.spectra[:Nneg, :Nneg]

        # Weights for integration
        weights = sel_dist(-self.neg_gammas, -self.neg_gammas, params)
        # Apply weighting. Could probably do this in a single fancy numpy
        # multiplication, which might be faster.
        weighted_spectra = 0*spectra
        for ii, row in enumerate(weights):
            for jj, w in enumerate(row):
                weighted_spectra[ii,jj] = w*spectra[ii,jj]

        # Integrate out the first two axes, which are the gamma axes.
        temp = np.trapz(weighted_spectra, -self.neg_gammas, axis=0)
        fs = np.trapz(temp, -self.neg_gammas, axis=0)

        if not exterior_int:
            return Spectrum(theta*fs)

        max_gamma = -self.neg_gammas[-1]
        min_gamma = -self.neg_gammas[0]
        # Account for density that is outside the simulated domain.
        weights_1low, weights_1high = 0*self.neg_gammas, 0*self.neg_gammas
        weights_2low, weights_2high = 0*self.neg_gammas, 0*self.neg_gammas
        for ii, gamma in enumerate(-self.neg_gammas):
            marg1_func = lambda g1: sel_dist(np.array([[g1]]),
                                             np.array([[gamma]]), params)
            marg2_func = lambda g2: sel_dist(np.array([[gamma]]),
                                             np.array([[g2]]), params)
            w_1low, err = scipy.integrate.quad(marg1_func, min_gamma, np.inf)
            w_1high, err = scipy.integrate.quad(marg1_func, 0, max_gamma)
            w_2low, err = scipy.integrate.quad(marg2_func, min_gamma, np.inf)
            w_2high, err = scipy.integrate.quad(marg2_func, 0, max_gamma)
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
        func = lambda g1,g2: sel_dist(np.array([[g1]]), np.array([[g2]]),
                                      params)
        weight, err = scipy.integrate.dblquad(func, 0, max_gamma,
                                              lambda _: 0, lambda _: max_gamma,
                                              epsrel=1e-3, epsabs=1e-4)
        fs += spectra[-1,-1]*weight

        # Neutral gamma2, strongly deleterious gamma1
        weight, err = scipy.integrate.dblquad(func, 0, max_gamma,
                                              lambda _: min_gamma,
                                              lambda _: np.inf,
                                              epsrel=1e-3, epsabs=1e-4)
        fs += spectra[0,-1]*weight

        # Neutral gamma1, strongly deleterious gamma2
        weight, err = scipy.integrate.dblquad(func, min_gamma, np.inf,
                                              lambda _: 0, lambda _: max_gamma,
                                              epsrel=1e-3, epsabs=1e-4)
        fs += spectra[-1,0]*weight

        return Spectrum(theta*fs)

    def integrate_biv_point_pos(self, params, ns, biv_seldist, theta,
                                rho=0, pts=None):
        """
        Integrate spectra over a bivariate prob. dist. for negative gammas plus
        a point mass of positive selection.

        params: Parameters for sel_dist and positive selection.
                It is assumed that the last four parameters are:
                Proportion positive selection in pop1, postive gamma for pop1,
                 prop. positive in pop2, and positive gamma for pop2.
                Earlier arguments are assumed to be for the continuous bivariate
                distribution.
        ns: Ignored
        biv_seldist: Bivariate probability distribution for negative selection,
                     taking in arguments (xx, yy, params)
        theta: Population-scaled mutation rate
        rho: Correlation coefficient used to connect negative and positive
             components of DFE.
        pts: Ignored

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
        p++ = p1*p2 + rho*(sqrt(p1*p2) - p1*p2)
        p+- = (1-rho) * p1*(1-p2)
        p-- = (1-p1)*(1-p2) + rho*(1-sqrt(p1*p2) - (1-p1)*(1-p2))

        The logic here is that in the rho=1 limit, we set the proportion
        positively selected to be the geometric mean of p1 and p2, the
        proportion positive in one pop and negative in the other (p+-) to be
        zero, and the remainder negative in both populations p--. We then
        linearly interpolate between the rho=0 and rho=1 cases.

        This requires the integration method to explicitly know about rho,
        so it's not completely general to all joint DFEs. rho is thus
        included as a parameter in the argument list.
        """
        biv_params = params[:-4]
        ppos1, gammapos1, ppos2, gammapos2 = params[-4:]

        self.check_normalization(biv_params, biv_seldist)

        Nneg = len(self.neg_gammas)
        weights = biv_seldist(-self.neg_gammas, -self.neg_gammas, biv_params)

        # Case in which both are negative, so we integrate directly
        # over the bivariate distribution
        neg_neg = self.integrate_biv(biv_params, ns, biv_seldist, 1, pts)

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
        pos_neg_weights = np.trapz(weights, -self.neg_gammas, axis=0)
        # For numpy multiplication...
        pos_neg_weights = pos_neg_weights[:,np.newaxis,np.newaxis]
        # Do the weighted integral.
        pos_neg = np.trapz(pos_neg_weights*pos_neg_spectra,
                           -self.neg_gammas, axis=0)

        # Case in which pop2 is positive and pop1 is negative.
        neg_pos_spectra = np.squeeze(self.spectra[:Nneg,self.gammas==gammapos2])
        # Now integrate over pop2 to get marginal DFE for pop1
        neg_pos_weights = np.trapz(weights, -self.neg_gammas, axis=1)
        neg_pos_weights = neg_pos_weights[:,np.newaxis,np.newaxis]
        neg_pos = np.trapz(neg_pos_weights*neg_pos_spectra,
                           -self.neg_gammas, axis=0)

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

    def integrate_biv_symmetric_point_pos(self, params, ns, biv_seldist, theta,
                                          rho=0, pts=None):
        """
        Convenience method for integrating spectra over a bivariate prob. dist.
        for negative gammas plus a symmetric point mass of positive selection.

        params: Parameters for sel_dist and positive selection.
                It is assumed that the last two parameters are:
                Proportion positive selection, postive gamma
                Earlier arguments are assumed to be for the continuous bivariate
                distribution.
                *It is assumed that the last of those earlier arguments is
                 the correlation coefficient rho.*
        ns: Ignored
        biv_seldist: Bivariate probability distribution for negative selection,
                     taking in arguments (xx, yy, params)
        theta: Population-scaled mutation rate
        rho: Correlation coefficient used to connect negative and positive
             components of DFE.
        pts: Ignored
        """
        seldist_params = params[:-2]
        rho = seldist_params[-1]
        ppos, gammapos = params[-2:]

        params = np.concatenate((seldist_params, [ppos,gammapos,ppos,gammapos]))

        return self.integrate_biv_point_pos(params, ns, biv_seldist, theta,
                                            rho=rho, pts=None)

    def integrate_mixture(self, params, ns, biv_seldist, theta, pts):
        """
        Weighted summation of 1d and 2d distributions that share parameters.
        The 1d distribution is equivalent to assuming selection coefficients are
        perfectly correlated.

        params: Parameters for potential optimization.
                It is assumed that last parameter is the weight for the 2d dist.
                The second-to-last parameter is assumed to be the correlation
                    coefficient for the 2d distribution.
                The remaining parameters as assumed to be shared between the
                    1d and 2d distributions.
        ns: Ignored
        biv_seldist: Bivariate probability distribution for s2
        theta: Population-scaled mutation rate
        pts: Ignored
        """
        fs1 = self.integrate_uni(params[:-2], ns, sel_dist2, theta, None)
        fs2 = self.integrate_biv(params[:-1], ns, sel_dist2, theta, None)

        p = params[-1]
        return (1-p)*fs1 + p*fs2

    def check_normalization(self, params, sel_dist):
        """
        Check normalization of continuous negative DFE based on gamma grid.

        Prints warning if normalization differs from 1 by more than
         self.normalization_threshold.
        """
        weights = sel_dist(-self.neg_gammas, -self.neg_gammas, params)
        norm = np.trapz(np.trapz(weights, -self.neg_gammas), -self.neg_gammas)
        if np.abs(norm - 1) > self.normalization_threshold:
            print('WARNING: Normalization of DFE with parameters {0} over '
                  'given gamma grid is {1}, which exceeds tolerance of {2}.'
                  .format(params, norm, self.normalization_threshold))
        return norm

#
# Probability distribution functions
#
def biv_lognorm_pdf(xx, yy, params):
    """
    Bivariate lognormal probability distribution function

    xx: x-coordinates at which to evaluate.
    yy: y-coordinates at which to evaluate.
    params: Input parameters. If len(params) == 3, then params = (mu,sigma,rho)
            and mu and sigma are assumed to be equal in the two dimensions. If
            len(params) == 5, then params = (mu1,mu2,sigma1,sigma2,rho)
    """
    if len(params) == 3:
        mu, sigma, rho = params
        mu1 = mu2 = mu
        sigma1 = sigma2 = sigma
    elif len(params) == 5:
        mu1, mu2, sigma1, sigma2, rho = params
    else:
        raise ValueError('Parameter array for bivariate lognormal must have '
                         'length 3 or 5.')
    delx = (np.log(xx[:,np.newaxis]) - mu1)/sigma1
    dely = (np.log(yy[np.newaxis,:]) - mu2)/sigma2
    norm = 2*np.pi * sigma1*sigma2 * np.sqrt(1.-rho**2) * np.outer(xx,yy)
    q = (delx**2 - 2.*rho*delx*dely + dely**2)/(1.-rho**2)

    return np.exp(-q/2.)/norm

def biv_ind_sym_gamma_pdf(xx, yy, params):
    """
    Bivariate gamma probability distribution function

    xx: x-coordinates at which to evaluate.
    yy: y-coordinates at which to evaluate.
    params: [alpha, beta, rho]

    As in Kibble (1941) and Smith and Adelfang (1981).
    """
    if len(params)==3:
        alpha, beta, rho = params
        alpha1 = alpha2 = alpha
        beta1 = beta2 = beta
    else:
        raise ValueError('Parameter array for bivariate independent '
                         'symmetric gamma must have length 3 or 5.')

    xmarg = scipy.stats.distributions.gamma.pdf(xx, alpha1, scale=beta1)
    ymarg = scipy.stats.distributions.gamma.pdf(yy, alpha2, scale=beta2)
    return np.outer(xmarg, ymarg)

#
# Plotting functions
#
def plot_biv_dfe(gammax, gammay, sel_dist, params, logweight=True, ax=None,
                 xlabel='$\gamma_1$', ylabel='$\gamma_2$', cmap='gray_r',
                 colorbar=True, vmin=0, clip_on=True):
    """
    Plot bivariate DFE (for negative gammas).

    gammax, gammay: Grids of gamma values to plot with. Note that non-negative
                    values are discarded.
    sel_dist: Bivariate probability distribution,
              taking in arguments (xx, yy, params)
    params: Parameters for sel_dist
    logweight: If True, plotted values are weighted by x and y, so that the
               total probability within each cell is plotted, rather than the
               probability density. This is typically easier to interpret.
    ax: Matplotlib axes to plot into. If None, plt.gca() is used.
    xlabel, ylabel: Labels for x and y axes.
    cmap: Colormap to plot with. For useful advice regarding colormaps, see
          http://matplotlib.org/users/colormaps.html .
    colorbar: If True, include a colorbar alongside the plot.
    vmin: Values below this will be colored white.
    clip_on: If False, plot will extend outside of axes.
    """
    # Discard non-negative values of gamma.
    gammax = gammax[gammax < 0]
    gammay = gammay[gammay < 0]

    # Midpoints of each gamma bin (on a log-scale)
    xmid = np.sqrt(gammax[:-1] * gammax[1:])
    ymid = np.sqrt(gammay[:-1] * gammay[1:])
    # Calculate weights
    weights = sel_dist(xmid, ymid, params)
    # Convert from probability density to probability value within each cell.
    if logweight:
        weights *= xmid[:,np.newaxis] * ymid[np.newaxis,:]

    # Plot data
    X,Y = np.meshgrid(gammax, gammay)

    if not ax:
        # Hiding import here, so Selection_2d.py isn't dependent on matplotlib.
        import matplotlib.pyplot as plt
        ax = plt.gca()
    masked = np.ma.masked_where(weights.T<vmin,weights.T)
    mappable = ax.pcolor(X,Y,masked, cmap=cmap, vmin=vmin, clip_on=clip_on)

    # Plot the colorbar, labeling carefully depending on logweight
    if colorbar:
        cb = ax.get_figure().colorbar(mappable)
        if logweight:
            cb.set_label('Probability')
        else:
            cb.set_label('Probability density')

    # Set to logscale. Use symlog because it handles negative values cleanly
    ax.set_xscale('symlog', linthreshx=abs(gammax).min())
    ax.set_yscale('symlog', linthreshy=abs(gammay).min())

    ax.set_xlabel(xlabel, fontsize='large')
    ax.set_ylabel(ylabel, fontsize='large')

    return ax

def plot_biv_point_pos_dfe(gammax, gammay, sel_dist, params, rho=0,
                           logweight=True, xlabel='$\gamma_1$',
                           ylabel='$\gamma_2$', cmap='gray_r', fignum=None,
                           colorbar=True, vmin=0):
    """
    Plot bivariate DFE (for negative gammas) with positive point mass.

    Returns figure for plot.

    Note: You might need to adjust subplot parameters using fig.subplots_adjust
          after plotting to see all labels, etc.

    gammax, gammay: Grids of gamma values to plot with. Note that non-negative
                    values are discarded.
    sel_dist: Bivariate probability distribution,
              taking in arguments (xx, yy, params)
    params: Parameters for sel_dist and positive selection.
            It is assumed that the last four parameters are:
            Proportion positive selection in pop1, postive gamma for pop1,
             prop. positive in pop2, and positive gamma for pop2.
            Earlier arguments are assumed to be for the continuous bivariate
            distribution.
    rho: Correlation coefficient used to connect negative and positive
         components of DFE.
    logweight: If True, plotted values are weighted by x and y, so that the
               total probability within each cell is plotted, rather than the
               probability density. This is typically easier to interpret.
    xlabel, ylabel: Labels for x and y axes.
    cmap: Colormap to plot with. For useful advice regarding colormaps, see
          http://matplotlib.org/users/colormaps.html .
    fignum: Figure number to use. If None or figure does not exist, a new
            one will be created.
    colorbar: If True, plot scale bar for probability
    vmin: Values below this will be colored white.
    """
    biv_params = params[:-4]
    ppos1, gammapos1, ppos2, gammapos2 = params[-4:]

    # Pull out negative gammas and calculate (log) midpoints.
    gammax = gammax[gammax < 0]
    gammay = gammay[gammay < 0]
    xmid = np.sqrt(gammax[:-1] * gammax[1:])
    ymid = np.sqrt(gammay[:-1] * gammay[1:])

    # Bivariate negative part of DFE
    neg_neg = sel_dist(xmid, ymid, biv_params)
    # Marginal DFEs from integrating along each axis
    neg_pos = np.trapz(neg_neg, ymid, axis=1)
    pos_neg = np.trapz(neg_neg, xmid, axis=0)

    if logweight:
        neg_neg *= xmid[:,np.newaxis] * ymid[np.newaxis,:]
        neg_pos *= xmid
        pos_neg *= ymid

    # Weighting factors for each quadrant of the DFE. Note that in the case
    # that ppos1==ppos2, these reduce to the model of Ragsdale et al. (2016)
    p_pos_pos = ppos1*ppos2 + rho*(np.sqrt(ppos1*ppos2) - ppos1*ppos2)
    p_pos_neg = (1-rho) * ppos1*(1-ppos2)
    p_neg_pos = (1-rho) * (1-ppos1)*ppos2
    p_neg_neg = (1-ppos1)*(1-ppos2)\
            + rho*(1-np.sqrt(ppos1*ppos2) - (1-ppos1)*(1-ppos2))

    # Apply weighting factors
    pos_neg *= p_pos_neg
    neg_pos *= p_neg_pos
    neg_neg *= p_neg_neg

    # For plotting, to put all on same color-scale.
    vmax = max([neg_neg.max(), neg_pos.max(), pos_neg.max(), p_pos_pos])

    # Grid points for plotting in positive selection regime.
    if gammapos1 != gammapos2:
        pos_grid = np.array(sorted([gammapos1, (gammapos1 + gammapos2)/2.,
                                    gammapos2]))
    else:
        pos_grid = np.array([gammapos1-1, gammapos1, gammapos1+1])

    # Fill in grids for plotting positive terms
    pos_neg_grid = np.zeros((3, len(ymid)))
    pos_neg_grid[pos_grid == gammapos1] = pos_neg
    neg_pos_grid = np.zeros((len(xmid), 3))
    neg_pos_grid[:,pos_grid == gammapos2] = neg_pos[:,np.newaxis]
    pos_pos_grid = np.zeros((3,3))
    pos_pos_grid[pos_grid == gammapos1, pos_grid == gammapos2] = p_pos_pos

    # For plotting with pcolor, need grid one size bigger.
    d = pos_grid[1] - pos_grid[0]
    plot_pos_grid = np.linspace(pos_grid[0] - d/2., pos_grid[-1] + d/2., 4)

    import matplotlib.pyplot as plt
    fig = plt.figure(num=fignum, figsize=(4,3), dpi=150)
    fig.clear()

    gs = plt.matplotlib.gridspec.GridSpec(2,3, width_ratios=[9,1,0.5],
                                          height_ratios=[1,9],
                                          wspace=0.15, hspace=0.15)
    # Create our four axes
    ax_ll = fig.add_subplot(gs[1,0])
    ax_ul = fig.add_subplot(gs[0,0], sharex=ax_ll)
    ax_lr = fig.add_subplot(gs[1,1], sharey=ax_ll)
    ax_ur = fig.add_subplot(gs[0,1], sharex=ax_lr, sharey=ax_ul)

    # Plot in each axis
    X,Y = np.meshgrid(gammax, gammay)
    masked = np.ma.masked_where(neg_neg.T<vmin,neg_neg.T)
    mappable = ax_ll.pcolor(X,Y,masked, cmap=cmap, vmin=vmin, vmax=vmax)
    X,Y = np.meshgrid(plot_pos_grid, plot_pos_grid)
    masked = np.ma.masked_where(pos_pos_grid.T<vmin,pos_pos_grid.T)
    ax_ur.pcolor(X,Y,masked, cmap=cmap, vmin=vmin, vmax=vmax)
    X,Y = np.meshgrid(plot_pos_grid, gammay)
    masked = np.ma.masked_where(pos_neg_grid.T<vmin,pos_neg_grid.T)
    ax_lr.pcolor(X,Y,masked, cmap=cmap, vmin=vmin, vmax=vmax)
    X,Y = np.meshgrid(gammax, plot_pos_grid)
    masked = np.ma.masked_where(neg_pos_grid.T<vmin,neg_pos_grid.T)
    ax_ul.pcolor(X,Y,masked, cmap=cmap, vmin=vmin, vmax=vmax)

    # Set logarithmic scale on legative parts
    ax_ll.set_xscale('symlog', linthreshx=abs(gammax).min())
    ax_ll.set_yscale('symlog', linthreshy=abs(gammay).min())

    # Remove interior axes lines and tickmarks
    ax_ll.spines['right'].set_visible(False)
    ax_ll.spines['top'].set_visible(False)
    ax_ll.tick_params('both', top=False, right=False, which='both')

    ax_ul.spines['right'].set_visible(False)
    ax_ul.spines['bottom'].set_visible(False)
    ax_ul.tick_params('both', top=True, bottom=False,
                      labelbottom=False, right=False, which='both')

    ax_lr.spines['left'].set_visible(False)
    ax_lr.spines['top'].set_visible(False)
    ax_lr.tick_params('both', right=True, left=False,
                      labelleft=False, top=False, which='both')

    ax_ur.spines['left'].set_visible(False)
    ax_ur.spines['bottom'].set_visible(False)
    ax_ur.tick_params('both', right=True, left=False,
                      labelleft=False, top=True, bottom=False,
                      labelbottom=False, which='both')

    # Set tickmarks for positive selection
    if gammapos1 != gammapos2:
        ax_ur.set_xticks([gammapos1, gammapos2])
        ax_ur.set_yticks([gammapos1, gammapos2])
    else:
        ax_ur.set_xticks([gammapos1])
        ax_ur.set_yticks([gammapos1])

    ax_ll.set_xlabel(xlabel, fontsize='large')
    ax_ll.set_ylabel(ylabel, fontsize='large')

    if colorbar:
        ax = fig.add_subplot(gs[1,2])
        cb = fig.colorbar(mappable, cax=ax, use_gridspec=True)
        if logweight:
            cb.set_label('Probability')
    fig.subplots_adjust(top=0.99, right=0.82, bottom=0.19, left=0.18)

    return fig

def plot_basic_mixture(gammas, sel_dist1, sel_dist2,
                       params, logweight=True, ax=None,
                       xlabel='$\gamma_1$', ylabel='$\gamma_2$', cmap='gray_r',
                       colorbar=True, vmin=0):
    """
    Plot bivariate DFE (for negative gammas).

    gammas: Grid of gamma values to plot with. Note that non-negative
            values are discarded.
    sel_dist1: Univariate probability distribution
    sel_dist2: Bivariate probability distribution
    params: Parameters for distributions
            It is assumed that last parameter is the weight for the 2d dist.
            The second-to-last parameter is assumed to be the correlation
                coefficient for the 2d distribution.
            The remaining parameters as assumed to be shared between the
                1d and 2d distributions.
    logweight: If True, plotted values are weighted by x and y, so that the
               total probability within each cell is plotted, rather than the
               probability density. This is typically easier to interpret.
    ax: Matplotlib axes to plot into. If None, plt.gca() is used.
    xlabel, ylabel: Labels for x and y axes.
    cmap: Colormap to plot with. For useful advice regarding colormaps, see
          http://matplotlib.org/users/colormaps.html .
    colorbar: If True, include a colorbar alongside the plot.
    vmin: Values below this will be colored white.
    """
    # Discard non-negative values of gamma.
    gammas = gammas[gammas < 0]

    # Midpoints of each gamma bin (on a log-scale)
    xmid = np.sqrt(gammas[:-1] * gammas[1:])
    # Calculate 2d weights
    weights2 = sel_dist2(xmid, xmid, params[:-1])

    # Calculate 1d weights and add in
    weights1 = np.eye(len(xmid)) * sel_dist1(-xmid, *params[:-2])

    # Convert from probability density to probability value within each cell.
    if logweight:
        weights2 *= xmid[:,np.newaxis] * xmid[np.newaxis,:]
        weights1 *= xmid[:,np.newaxis]

    p = params[-1]
    weights = (1-p)*weights1 + p*weights2

    # Plot data
    X,Y = np.meshgrid(gammas, gammas)

    if not ax:
        # Hiding import here, so Selection_2d.py isn't dependent on matplotlib.
        import matplotlib.pyplot as plt
        ax = plt.gca()
    masked = np.ma.masked_where(weights.T<vmin,weights.T)
    mappable = ax.pcolor(X,Y,masked, cmap=cmap, vmin=vmin)

    # Plot the colorbar, labeling carefully depending on logweight
    if colorbar:
        cb = ax.get_figure().colorbar(mappable)
        if logweight:
            cb.set_label('Probability')
        else:
            cb.set_label('Probability density')

    # Set to logscale. Use symlog because it handles negative values cleanly
    ax.set_xscale('symlog', linthreshx=abs(gammas).min())
    ax.set_yscale('symlog', linthreshy=abs(gammas).min())

    ax.set_xlabel(xlabel, fontsize='large')
    ax.set_ylabel(ylabel, fontsize='large')

    return ax

#
# Example demography + selection functions
#
import dadi
from dadi import Numerics, PhiManip, Integration
from dadi.Spectrum_mod import Spectrum

def trivial_fs(params, ns, pts):
    """
    For testing.
    """
    return Spectrum([1], mask_corners=False)

def IM_sel(params, ns, pts):
    s,nu1,nu2,T,m12,m21,gamma1,gamma2 = params

    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx, gamma=gamma1)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    nu1_func = lambda t: s * (nu1/s)**(t/T)
    nu2_func = lambda t: (1-s) * (nu2/(1-s))**(t/T)
    phi = Integration.two_pops(phi, xx, T, nu1_func, nu2_func,
                               m12=m12, m21=m21, gamma1=gamma1,
                               gamma2=gamma2)

    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs

def IM_single_sel(params, ns, pts):
    s,nu1,nu2,T,m12,m21,gamma = params
    return IM_sel([s,nu1,nu2,T,m12,m21,gamma,gamma], ns, pts)

def split_mig_sel(params, ns, pts):
    nu1,nu2,T,m,gamma1,gamma2 = params

    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx, gamma=gamma1)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    phi = Integration.two_pops(phi, xx, T, nu1, nu2, m12=m, m21=m, gamma1=gamma1,
                               gamma2=gamma2)

    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs

def split_mig_single_sel(params, ns, pts):
    nu1,nu2,T,m,gamma1,gamma2 = params
    return split_mig_sel([nu1,nu2,T,m,gamma,gamma], ns, pts)

def IM_pre_sel(params, ns, pts):
    nuPre,TPre,s,nu1,nu2,T,m12,m21,gamma1,gamma2 = params

    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx, gamma=gamma1)
    phi = Integration.one_pop(phi, xx, TPre, nu=nuPre, gamma=gamma1)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    nu1_0 = nuPre*s
    nu2_0 = nuPre*(1-s)
    nu1_func = lambda t: nu1_0 * (nu1/nu1_0)**(t/T)
    nu2_func = lambda t: nu2_0 * (nu2/nu2_0)**(t/T)
    phi = Integration.two_pops(phi, xx, T, nu1_func, nu2_func,
                               m12=m12, m21=m21, gamma1=gamma1,
                               gamma2=gamma2)

    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs

def IM_pre_single_sel(params, ns, pts):
    nuPre,TPre,s,nu1,nu2,T,m12,m21,gamma = params
    return IM_pre_sel([nuPre,TPre,s,nu1,nu2,T,m12,m21,gamma,gamma], ns, pts)

#
#
#

def mixture_basic(params, ns, s1, s2, sel_dist1, sel_dist2, theta, pts,
                  exterior_int=True):
    """
    Weighted summation of 1d and 2d distributions that share parameters.
    The 1d distribution is equivalent to assuming selection coefficients are
    perfectly correlated.

    params: Parameters for potential optimization.
            It is assumed that last parameter is the weight for the 2d dist.
            The second-to-last parameter is assumed to be the correlation
                coefficient for the 2d distribution.
            The remaining parameters as assumed to be shared between the
                1d and 2d distributions.
    ns: Ignored
    s1: spectra object for 1d distribution
    s2: spectra2d object for 2d distribution
    sel_dist1: Univariate probability distribution for s1
    sel_dist2: Bivariate probability distribution for s2
    theta: Population-scaled mutation rate
    pts: Ignored
    exterior_int: If False, do not integrate outside sampled domain.
    """
    fs1 = s1.integrate(params[:-2], sel_dist1, theta, exterior_int)
    fs2 = s2.integrate_biv(params[:-1], ns, sel_dist2, theta, None, exterior_int)

    p = params[-1]
    return (1-p)*fs1 + p*fs2

def mixture_symmetric_point_pos(params, ns, s1, s2, sel_dist1, sel_dist2,
                                theta, func_ex, pts=None):
    """
    Weighted summation of 1d and 2d distributions with positive selection.
    The 1d distribution is equivalent to assuming selection coefficients are
    perfectly correlated.

    params: Parameters for potential optimization.
            The last parameter is the weight for the 2d dist.
            The second-to-last parameter is positive gammma for the point mass.
            The third-to-last parameter is the proportion of positive selection.
            The fourth-to-last parameter is the correlation coefficient for the
                2d distribution.
            The remaining parameters as must be shared between the 1d and 2d
                distributions.
    ns: Ignored
    s1: spectra object for 1d distribution
    s2: spectra2d object for 2d distribution
    sel_dist1: Univariate probability distribution for s1
    sel_dist2: Bivariate probability distribution for s2
    theta: Population-scaled mutation rate
    func_ex: dadi function for simulating model with a single selection
             coefficient, as in s1.integrate_point_pos
    pts: Ignored
    """
    pdf_params = params[:-4]
    rho, ppos, gamma_pos, p2d = params[-4:]

    params1 = list(pdf_params) + [ppos, gamma_pos]
    fs1 = s1.integrate_point_pos(params1, sel_dist1, theta, func_ex)
    params2 = list(pdf_params) + [rho, ppos, gamma_pos]
    fs2 = s2.integrate_biv_symmetric_point_pos(params2, ns, sel_dist2,
                                               theta, rho=rho)
    return (1-p2d)*fs1 + p2d*fs2

if __name__ == '__main__':
    import cPickle, random
    import matplotlib.pyplot as plt

    # Sanity check on the alignment of the axes of the lognormal distribution.
    # Params create lognormal that is very wide for population 2.
    params = [0,0,0.1,10,0]
    xx = np.logspace(-2,2,21)
    output = biv_lognorm_pdf(xx,xx,params)
    # Corresponds to a fixed value in pop1, and all values in pop2
    print("Should be small : {0}".format(np.std(output[10]*xx)))
    # Corresponds to a fixed value in pop2, and all values in pop1
    print("Should be larger: {0}".format(np.std(output[:,10]*xx)))

    # Example of plotting a joint DFE
    sel_dist = biv_lognorm_pdf
    params = [0.5,-0.5,0.5,1,-0.8]
    gammax = -np.logspace(-2, 1, 20)
    gammay = -np.logspace(-1, 2, 30)

    fig = plt.figure(137, figsize=(4,3), dpi=150)
    fig.clear()
    ax = fig.add_subplot(1,1,1)
    plot_biv_dfe(gammax, gammay, sel_dist, params, logweight=True, ax=ax)
    fig.tight_layout()

    # With positive selection
    sel_dist = biv_lognorm_pdf
    params = [0.5,-0.5,0.5,1,0.0,0.3,3,0.3,4]
    gammax = -np.logspace(-2, 1, 20)
    gammay = -np.logspace(-1, 2, 21)
    fig = plot_biv_point_pos_dfe(gammax, gammay, sel_dist, params,
                                 fignum=23, rho=params[4])

    #
    # Trivial test of normalization.
    #
    s = spectra2d([], [], trivial_fs, pts=1, Npts=100,
                  int_bounds=(1e-4, 1000.))
    params = [2,1,0]
    # Also check warning about normalization threshold
    print('Should see warning about normalization, with this strict threshold.')
    s.normalization_threshold = 0.001
    output = s.integrate_biv(params, None, biv_lognorm_pdf, 1, None)
    print('Normalization test: {0:.4f}'.format(output.data[0]))

    #
    # Full test of optimization machinery. 
    # Considering only a narrow range of gammas, so integration is faster.
    #
    demo_params = [0.5,2,0.5,0.1,0,0]
    ns = [8, 12]
    pts_l = [60, 80, 100]
    func_ex = dadi.Numerics.make_extrap_func(IM_sel)
    # Check whether we already have a chached set of 2d spectra. If not
    # generate them.
    try:
        s = cPickle.load(file('test.spectra2d.bpkl'))
    except IOError:
        s = spectra2d(demo_params, ns, func_ex, pts=pts_l, Npts=100,
                      int_bounds=(1e-2,10), echo=True, mp=True,
                      additional_gammas=[1.2,4.3])
        # Save spectra2d object
        fid = file('test.spectra2d.bpkl', 'w')
        cPickle.dump(s, fid, protocol=2)
        fid.close()

    # Set normalizaton threshold to reasonable value.
    s.normalization_threshold = 0.1

    # Generate test data set to fit
    input_params, theta = [0.5,0.5,-0.8], 1e5
    sel_dist = biv_lognorm_pdf
    # Expected sfs
    target = s.integrate_biv(input_params, None, sel_dist, theta, None)
    # Seed random number generator, so test is reproducible
    np.random.seed(1398238)
    # Get data with Poisson variance around expectation
    data = target.sample()

    p0 = [0,1.,0.8]
    popt = dadi.Inference.optimize(p0, data, s.integrate_biv, pts=None,
                                   func_args=[sel_dist, theta],
                                   lower_bound=[None,0,-1],
                                   upper_bound=[None,None,1],
                                   verbose=30, multinom=False)
    print('Input parameters: {0}'.format(input_params))
    print('Optimized parameters: {0}'.format(popt))

    # Plot inferred DFE. Note that this will render slowly, because grid of
    # gammas is fairly dense.
    fig = plt.figure(231, figsize=(4,3), dpi=150)
    fig.clear()
    ax = fig.add_subplot(1,1,1)
    plot_biv_dfe(s.gammas, s.gammas, sel_dist, popt, ax=ax)
    fig.tight_layout()

    #
    # Test point mass of positive selection. To do so, we test against
    # the single-population case using very high correlation.
    #
    params = [-0.5,0.5,0.99,0.1, 4.3, 0.1, 4.3]
    fs_biv = s.integrate_biv_point_pos(params, None, sel_dist, theta,
                                       rho=params[2], pts=None)
    norm = s.check_normalization(params[:3], sel_dist)
    print('Normalization for 2D test: {0}'.format(norm))

    func_single_ex = dadi.Numerics.make_extrap_func(IM_single_sel)
    import Selection
    try:
        s1 = cPickle.load(file('test.spectra1d.bpkl'))
    except IOError:
        s1 = Selection.spectra(demo_params, ns, func_single_ex, pts_l=pts_l,
                               Npts=100, int_bounds=(1e-2, 10), mp=False,
                               echo=False)
        fid = file('test.spectra1d.bpkl', 'w')
        cPickle.dump(s1, fid, protocol=2)
        fid.close()

    fs1 = s1.integrate_point_pos([-0.5,0.5,0.1,4.3], Selection.lognormal_dist,
                                 1e5, func_single_ex)

    fig = dadi.Plotting.pylab.figure(229)
    fig.clear()
    dadi.Plotting.plot_2d_comp_Poisson(fs1, fs_biv)

    #
    # Test optimization of point mass positive selection. First, test against
    # case of asymmetric positive selection fit with symmetric.
    #

    # Generate test data set to fit
    # This is a symmetric case, with mu1=mu2=0.5, sigma1=sigma2=0.3, rho=-0.5,
    # ppos1=ppos2=0.2, gammapos1=gammapos2=1.2.
    input_params, theta = [0.5,0.5,0.3,0.3,-0.5,0.2,1.2,0.2,1.2], 1e5
    # Expected sfs
    target = s.integrate_biv_point_pos(input_params, None, sel_dist, theta,
                                       rho=input_params[4], pts=None)
    # Get data with Poisson variance around expectation
    data = target.sample()

    # Now we'll fit using our special-case symmetric function. The last
    # two arguments are ppos and gammapos. The first three are thus for the
    # lognormal distribution. Note that our lognormal distribution assumes
    # symmetry if the length of the arguments is only three. If we wanted
    # asymmetric lognormal, we would pass in a p0 of total length 7.
    p0 = [0.3,0.3,0.1,0.2,1.2]
    popt = dadi.Inference.optimize(p0, data,
                                   s.integrate_biv_symmetric_point_pos,
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
                                   verbose=30, multinom=False)
    print('Symmetric test fit')
    print('  Input parameters: {0}'.format(input_params))
    print('  Optimized parameters: {0}'.format(popt))

    input_params, theta = [-0.5,0.5,1.3,0.3,0.2,0.1,1.2,0.2,4.3], 1e5
    target = s.integrate_biv_point_pos(input_params, None, sel_dist, theta,
                                       rho=input_params[4], pts=None)
    # Get data with Poisson variance around expectation
    data = target.sample()
    popt = dadi.Inference.optimize(p0, data,
                                   s.integrate_biv_symmetric_point_pos,
                                   pts=None, func_args=[sel_dist, theta],
                                   lower_bound=[-1,0.1,-1,0,0],
                                   upper_bound=[1,1,1,1,None],
                                   fixed_params=[None,None,None,None,1.2],
                                   verbose=30, multinom=False)
    print('Assymmetric test fit')
    print('  Input parameters: {0}'.format(input_params))
    print('  Optimized parameters: {0}'.format(popt))

    #
    # Test Godambe code for estimating uncertainties
    #
    input_params = [0.3,0.3,0.1,0.2,1.2]
    # Generate data in segments for future bootstrapping
    fs0 = s.integrate_biv_symmetric_point_pos(input_params, None, sel_dist,
                                              theta/100., rho=input_params[2],
                                              pts=None)
    # The multiplication of fs0 is to create a range of data size among
    # bootstrap chunks, which creates a range of thetas in the bootstrap
    # data sets.
    data_pieces = [(fs0*(0.5 + (1.5-0.5)/99*ii)).sample() for ii in range(100)]
    # Add up those segments to get our data spectrum
    data = dadi.Spectrum(np.sum(data_pieces, axis=0))
    # Do the optimization
    popt = dadi.Inference.optimize([0.2,0.2,0.15,0.3,1.2], data,
                                   s.integrate_biv_symmetric_point_pos,
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
        return s.integrate_biv_symmetric_point_pos(params, None, sel_dist,
                                                   theta, rho=params[2],
                                                   pts=None)

    # Run the uncertainty analysis
    uncerts = dadi.Godambe.GIM_uncert(temp_func, [], all_boot, popt[:-1], data,
                                      multinom=False)
    print('Godambe uncertainty test')
    print('  Input parameters: {0}'.format(input_params))
    print('  Optimized parameters: {0}'.format(popt))
    print('  Estimated 95% uncertainties      : {0}'.format(1.96*uncerts))

    boot_theta_adjusts = [b.sum()/data.sum() for b in all_boot]
    uncerts_adj = dadi.Godambe.GIM_uncert(temp_func, [], all_boot, popt[:-1],
                                          data, multinom=False,
                                          boot_theta_adjusts=boot_theta_adjusts)
    print('  Estimated 95% uncerts (theta adj): {0}'
          .format(1.96*uncerts_adj))

    # Ensure plots show up on screen.
    plt.show()
