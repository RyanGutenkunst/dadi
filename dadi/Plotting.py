"""
Routines for Plotting comparisons between model and data.

These can serve as inspiration for custom routines for one's own purposes.
"""

import matplotlib
import pylab
import numpy

log10_m = numpy.ma.masked_unary_operation(numpy.log10)

import Numerics, SFS

def plot_1d_comp_multinom(model, data, fig_num=None):
    """
    Mulitnomial comparison between 1d model and data.


    model: 1-dimensional model SFS
    data: 1-dimensional data SFS
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.

    This comparison is multinomial in that it rescales the model to optimally
    fit the data.
    """
    if fig_num is None:
        f = pylab.gcf()
    else:
        f = pylab.figure(fig_num, figsize=(7,7))
    pylab.clf()

    masked_model, masked_data = Numerics.intersect_masks(model, data)
    masked_model = SFS.optimally_scaled_sfs(masked_model, masked_data)

    ax = pylab.subplot(2,1,1)
    pylab.semilogy(masked_data, '-ob')
    pylab.semilogy(masked_model, '-or')

    pylab.subplot(2,1,2, sharex = ax)
    resid = SFS.Anscombe_Poisson_residual(masked_model, masked_data)
    pylab.plot(resid, '-og')

    ax.set_xlim(0, data.shape[0]-1)
    pylab.show()

def plot_single_2d_sfs(sfs, vmin=None, vmax=None, ax=None, 
                       pop1_label = 'pop1', pop2_label='pop2'):
    """
    Logarithmic heatmap of single 2d SFS.

    sfs: SFS to plot
    vmin: Values in sfs below 10**vmin are masked in plot.
    vmax: Values in sfs above 10**vmin saturate the color spectrum.
    ax: Axes object to plot into. If None, the result of pylab.gca() is used.
    pop1_label: Label for population 1.
    pop2_label: Label for population 2.
    """
    if ax is None:
        ax = pylab.gca()

    if vmin is None:
        vmin = sfs.min()
    if vmax is None:
        vmax = sfs.max()

    mappable=ax.pcolor(numpy.ma.masked_where(sfs<vmin, sfs), 
                       cmap=pylab.cm.hsv, vmax=vmax, vmin=vmin, shading='flat',
                       norm = matplotlib.colors.LogNorm())
    #cbticks = [numpy.round(vmin+0.05,1), numpy.round(vmax-0.05, 1)]

    # This can be passed to colorbar (format=format) to make ticks be 10^blah.
    # But I don't think it looks particularly nice.
    format = matplotlib.ticker.FormatStrFormatter('$10^{%.1f}$')
    ax.figure.colorbar(mappable)

    ax.plot([0,sfs.shape[1]],[0, sfs.shape[0]], '-k', lw=0.2)

    ax.set_ylabel(pop1_label, horizontalalignment='left')
    ax.set_xlabel(pop2_label, verticalalignment='bottom')
    ax.set_xticks([0.5, sfs.shape[1]-0.5])
    ax.set_xticklabels([str(0), str(sfs.shape[1]-1)])
    ax.set_yticks([0.5, sfs.shape[0]-0.5])
    ax.set_yticklabels([str(0), str(sfs.shape[0]-1)])
    for tick in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
        tick.set_visible(False)
    ax.set_xlim(0, sfs.shape[1])
    ax.set_ylim(0, sfs.shape[0])

def plot_2d_resid(resid, resid_range=3, ax=None, 
                  pop1_label = 'pop1', pop2_label='pop2'):
    """
    Linear heatmap of 2d residual array.

    sfs: Residual array to plot.
    resid_range: Values > resid range or < resid_range saturate the color
                 spectrum.
    ax: Axes object to plot into. If None, the result of pylab.gca() is used.
    pop1_label: Label for population 1.
    pop2_label: Label for population 2.
    """
    if ax is None:
        ax = pylab.gca()

    if resid_range is None:
        resid_range = abs(resid).max()

    mappable=ax.pcolor(resid, cmap=pylab.cm.RdBu_r, vmin=-resid_range, 
                       vmax=resid_range, shading='flat')

    cbticks = [-resid_range, 0, resid_range]
    # This can be passed to colorbar (format=format) to make ticks be 10^blah.
    # But I don't think it looks particularly nice.
    format = matplotlib.ticker.FormatStrFormatter('$10^{%.1f}$')
    ax.figure.colorbar(mappable, ticks=cbticks)

    ax.plot([0,resid.shape[1]],[0, resid.shape[0]], '-k', lw=0.2)

    ax.set_ylabel(pop1_label, horizontalalignment='left')
    ax.set_xlabel(pop2_label, verticalalignment='bottom')
    ax.set_xticks([0.5, resid.shape[1]-0.5])
    ax.set_xticklabels([str(0), str(resid.shape[1]-1)])
    ax.set_yticks([0.5, resid.shape[0]-0.5])
    ax.set_yticklabels([str(0), str(resid.shape[0]-1)])
    for tick in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
        tick.set_visible(False)
    ax.set_xlim(0, resid.shape[1])
    ax.set_ylim(0, resid.shape[0])

def plot_2d_comp_multinom(model, data, vmin=None, vmax=None,
                          resid_range=3, fig_num=None,
                          pop1_label = 'pop1', pop2_label='pop2'):
    """
    Mulitnomial comparison between 2d model and data.


    model: 1-dimensional model SFS
    data: 1-dimensional data SFS
    vmin, vmax: Minimum and maximum values plotted for sfs are 10**vmin and
                10**vmax respectively.
    resid_range: Residual plot saturates at +- resid_range.
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.
    pop1_label: Label for population 1.
    pop2_label: Label for population 2.

    This comparison is multinomial in that it rescales the model to optimally
    fit the data.
    """
    if fig_num is None:
        f = pylab.gcf()
    else:
        f = pylab.figure(fig_num, figsize=(7,7))

    pylab.clf()
    pylab.subplots_adjust(bottom=0.07, left=0.07, top=0.95, right=0.95)

    masked_model, masked_data = Numerics.intersect_masks(model, data)
    masked_model = SFS.optimally_scaled_sfs(masked_model, masked_data)

    if vmax is None:
        vmax = max(model.max(), data.max())
    if vmin is None:
        vmin = min(model.min(), data.min())

    ax = pylab.subplot(2,2,1)
    plot_single_2d_sfs(masked_data, vmin=vmin, vmax=vmax,
                       pop1_label=pop1_label, pop2_label=pop2_label)

    pylab.subplot(2,2,2, sharex=ax, sharey=ax)
    plot_single_2d_sfs(masked_model, vmin=vmin, vmax=vmax,
                       pop1_label=pop1_label, pop2_label=pop2_label)

    resid = SFS.Anscombe_Poisson_residual(masked_model, masked_data,
                                          mask=vmin)
    pylab.subplot(2,2,3, sharex=ax, sharey=ax)
    plot_2d_resid(resid, resid_range, 
                  pop1_label=pop1_label, pop2_label=pop2_label)

    ax = pylab.subplot(2,2,4)
    flatresid = numpy.compress(numpy.logical_not(resid.mask.flat), resid.flat)
    ax.hist(flatresid, bins=20, normed=True)
    ax.set_yticks([])
    pylab.show()
