"""
Miscellaneous functions for plotting DFEs
"""
import numpy as np

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
    ax.set_xscale('symlog', linthresh=abs(gammax).min())
    ax.set_yscale('symlog', linthresh=abs(gammay).min())

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
    ax_ll.set_xscale('symlog', linthresh=abs(gammax).min())
    ax_ll.set_yscale('symlog', linthresh=abs(gammay).min())

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
