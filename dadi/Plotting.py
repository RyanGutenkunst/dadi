"""
Routines for Plotting comparisons between model and data.

These can serve as inspiration for custom routines for one's own purposes.
"""

import matplotlib
import pylab
import numpy

log10_m = numpy.ma.masked_unary_operation(numpy.log10)

# Together these define a custom set of ticks that labels only the lowest and
# highest bins visible in an SFS plot. These adjust nicely when zooming around. 
class sfsTickLocator(matplotlib.ticker.Locator):
    def __call__(self):
        'Return the locations of the ticks'

        self.verify_intervals()
        vmin, vmax = self.viewInterval.get_bounds()
        dmin, dmax = self.dataInterval.get_bounds()

        tmin = max(vmin, dmin)
        tmax = min(vmax, dmax)

        return numpy.array([round(tmin)+0.5, round(tmax)-0.5])
ctf = matplotlib.ticker.FuncFormatter(lambda x,pos: '%i' % (x-0.4))


import Numerics, SFS

def plot_1d_comp_multinom(model, data, fig_num=None, residual='Anscombe'):
    """
    Mulitnomial comparison between 1d model and data.


    model: 1-dimensional model SFS
    data: 1-dimensional data SFS
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.
    residual: 'Anscombe' for Anscombe residuals, which are more normally
              distributed for Poisson sampling. 'linear' for the linear
              residuals, which can be less biased.

    This comparison is multinomial in that it rescales the model to optimally
    fit the data.
    """
    masked_model, masked_data = Numerics.intersect_masks(model, data)
    masked_model = SFS.optimally_scaled_sfs(masked_model, masked_data)
    plot_1d_comp_Poisson(masked_model, masked_data, fig_num, residual)

def plot_1d_comp_Poisson(model, data, fig_num=None, residual='Anscombe'):
    """
    Poisson comparison between 1d model and data.


    model: 1-dimensional model SFS
    data: 1-dimensional data SFS
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.
    residual: 'Anscombe' for Anscombe residuals, which are more normally
              distributed for Poisson sampling. 'linear' for the linear
              residuals, which can be less biased.
    """
    if fig_num is None:
        f = pylab.gcf()
    else:
        f = pylab.figure(fig_num, figsize=(7,7))
    pylab.clf()

    masked_model, masked_data = Numerics.intersect_masks(model, data)

    ax = pylab.subplot(2,1,1)
    pylab.semilogy(masked_data, '-ob')
    pylab.semilogy(masked_model, '-or')

    pylab.subplot(2,1,2, sharex = ax)
    if residual == 'Anscombe':
        resid = SFS.Anscombe_Poisson_residual(masked_model, masked_data)
    elif residual == 'linear':
        resid = SFS.linear_Poisson_residual(masked_model, masked_data)
    else:
        raise ValueError("Unknown class of residual '%s'." % residual)
    pylab.plot(resid, '-og')

    ax.set_xlim(0, data.shape[0]-1)
    pylab.show()

def plot_single_2d_sfs(sfs, vmin=None, vmax=None, ax=None, 
                       pop1_label= 'pop1', pop2_label='pop2'):
    """
    Logarithmic heatmap of single 2d SFS.

    sfs: SFS to plot
    vmin: Values in sfs below vmin are masked in plot.
    vmax: Values in sfs above vmin saturate the color spectrum.
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

    ax.xaxis.set_major_formatter(ctf)
    ax.xaxis.set_major_locator(sfsTickLocator())
    ax.yaxis.set_major_formatter(ctf)
    ax.yaxis.set_major_locator(sfsTickLocator())
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

    ax.xaxis.set_major_formatter(ctf)
    ax.xaxis.set_major_locator(sfsTickLocator())
    ax.yaxis.set_major_formatter(ctf)
    ax.yaxis.set_major_locator(sfsTickLocator())
    for tick in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
        tick.set_visible(False)

    ax.set_xlim(0, resid.shape[1])
    ax.set_ylim(0, resid.shape[0])


def plot_2d_comp_multinom(model, data, vmin=None, vmax=None,
                          resid_range=3, fig_num=None,
                          pop1_label='pop1', pop2_label='pop2',
                          residual='Anscombe'):
    """
    Mulitnomial comparison between 2d model and data.


    model: 2-dimensional model SFS
    data: 2-dimensional data SFS
    vmin, vmax: Minimum and maximum values plotted for sfs are vmin and
                vmax respectively.
    resid_range: Residual plot saturates at +- resid_range.
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.
    pop1_label: Label for population 1.
    pop2_label: Label for population 2.
    residual: 'Anscombe' for Anscombe residuals, which are more normally
              distributed for Poisson sampling. 'linear' for the linear
              residuals, which can be less biased.

    This comparison is multinomial in that it rescales the model to optimally
    fit the data.
    """
    masked_model, masked_data = Numerics.intersect_masks(model, data)
    masked_model = SFS.optimally_scaled_sfs(masked_model, masked_data)

    plot_2d_comp_Poisson(masked_model, masked_data, vmin=vmin, vmax=vmax,
                         resid_range=resid_range, fig_num=fig_num,
                         pop1_label=pop1_label, pop2_label=pop2_label)
    
def plot_2d_comp_Poisson(model, data, vmin=None, vmax=None,
                         resid_range=3, fig_num=None,
                         pop1_label='pop1', pop2_label='pop2',
                         residual='Anscombe'):
    """
    Poisson comparison between 2d model and data.


    model: 2-dimensional model SFS
    data: 2-dimensional data SFS
    vmin, vmax: Minimum and maximum values plotted for sfs are vmin and
                vmax respectively.
    resid_range: Residual plot saturates at +- resid_range.
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.
    pop1_label: Label for population 1.
    pop2_label: Label for population 2.
    residual: 'Anscombe' for Anscombe residuals, which are more normally
              distributed for Poisson sampling. 'linear' for the linear
              residuals, which can be less biased.
    """
    masked_model, masked_data = Numerics.intersect_masks(model, data)

    if fig_num is None:
        f = pylab.gcf()
    else:
        f = pylab.figure(fig_num, figsize=(7,7))

    pylab.clf()
    pylab.subplots_adjust(bottom=0.07, left=0.07, top=0.95, right=0.95)

    if vmax is None:
        vmax = max(masked_model.max(), masked_data.max())
    if vmin is None:
        vmin = min(masked_model.min(), masked_data.min())

    ax = pylab.subplot(2,2,1)
    plot_single_2d_sfs(masked_data, vmin=vmin, vmax=vmax,
                       pop1_label=pop1_label, pop2_label=pop2_label)

    pylab.subplot(2,2,2, sharex=ax, sharey=ax)
    plot_single_2d_sfs(masked_model, vmin=vmin, vmax=vmax,
                       pop1_label=pop1_label, pop2_label=pop2_label)

    if residual == 'Anscombe':
        resid = SFS.Anscombe_Poisson_residual(masked_model, masked_data,
                                              mask=vmin)
    elif residual == 'linear':
        resid = SFS.linear_Poisson_residual(masked_model, masked_data,
                                            mask=vmin)
    else:
        raise ValueError("Unknown class of residual '%s'." % residual)

    pylab.subplot(2,2,3, sharex=ax, sharey=ax)
    plot_2d_resid(resid, resid_range, 
                  pop1_label=pop1_label, pop2_label=pop2_label)

    ax = pylab.subplot(2,2,4)
    flatresid = numpy.compress(numpy.logical_not(resid.mask.flat), resid.flat)
    ax.hist(flatresid, bins=20, normed=True)
    ax.set_yticks([])
    pylab.show()

def plot_3d_comp_multinom(model, data, vmin=None, vmax=None,
                          resid_range=3, fig_num=None,
                          pop1_label='pop1', pop2_label='pop2',
                          pop3_label='pop3', residual='Anscombe'):
    """
    Multinomial comparison between 3d model and data.


    model: 3-dimensional model SFS
    data: 3-dimensional data SFS
    vmin, vmax: Minimum and maximum values plotted for sfs are vmin and
                vmax respectively.
    resid_range: Residual plot saturates at +- resid_range.
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.
    pop1_label: Label for population 1.
    pop2_label: Label for population 2.
    pop3_label: Label for population 3.
    residual: 'Anscombe' for Anscombe residuals, which are more normally
              distributed for Poisson sampling. 'linear' for the linear
              residuals, which can be less biased.

    This comparison is multinomial in that it rescales the model to optimally
    fit the data.
    """
    masked_model, masked_data = Numerics.intersect_masks(model, data)
    masked_model = SFS.optimally_scaled_sfs(masked_model, masked_data)

    plot_3d_comp_Poisson(masked_model, masked_data, vmin=vmin, vmax=vmax,
                         resid_range=resid_range, fig_num=fig_num,
                         pop1_label=pop1_label, pop2_label=pop2_label,
                         pop3_label=pop3_label)

def plot_3d_comp_Poisson(model, data, vmin=None, vmax=None,
                         resid_range=3, fig_num=None,
                         pop1_label='pop1', pop2_label='pop2',
                         pop3_label='pop3', residual='Anscombe'):
    """
    Poisson comparison between 3d model and data.


    model: 3-dimensional model SFS
    data: 3-dimensional data SFS
    vmin, vmax: Minimum and maximum values plotted for sfs are vmin and
                vmax respectively.
    resid_range: Residual plot saturates at +- resid_range.
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.
    pop1_label: Label for population 1.
    pop2_label: Label for population 2.
    pop3_label: Label for population 3.
    residual: 'Anscombe' for Anscombe residuals, which are more normally
              distributed for Poisson sampling. 'linear' for the linear
              residuals, which can be less biased.
    """
    masked_model, masked_data = Numerics.intersect_masks(model, data)

    if fig_num is None:
        f = pylab.gcf()
    else:
        f = pylab.figure(fig_num, figsize=(8,10))

    pylab.clf()
    pylab.subplots_adjust(bottom=0.07, left=0.07, top=0.95, right=0.95)

    if vmax is None:
        modelmax = max(masked_model.sum(axis=sax).max() for sax in range(3))
        datamax = max(masked_data.sum(axis=sax).max() for sax in range(3))
        vmax = max(modelmax, datamax)
    if vmin is None:
        modelmin = min(masked_model.sum(axis=sax).min() for sax in range(3))
        datamin = min(masked_data.sum(axis=sax).min() for sax in range(3))
        vmin = min(modelmin, datamin)

    pop_labels = [pop1_label, pop2_label, pop3_label]

    for sax in range(3):
        marg_data = masked_data.sum(axis=2-sax)
        marg_model = masked_model.sum(axis=2-sax)

        labels = pop_labels[:]
        del labels[2-sax]

        ax = pylab.subplot(4,3,sax+1)
        plot_single_2d_sfs(marg_data, vmin=vmin, vmax=vmax,
                           pop1_label=labels[0], pop2_label=labels[1])

        pylab.subplot(4,3,sax+4, sharex=ax, sharey=ax)
        plot_single_2d_sfs(marg_model, vmin=vmin, vmax=vmax,
                           pop1_label=labels[0], pop2_label=labels[1])

        if residual == 'Anscombe':
            resid = SFS.Anscombe_Poisson_residual(marg_model, marg_data,
                                                  mask=vmin)
        elif residual == 'linear':
            resid = SFS.linear_Poisson_residual(marg_model, marg_data,
                                                mask=vmin)
        else:
            raise ValueError("Unknown class of residual '%s'." % residual)
        pylab.subplot(4,3,sax+7, sharex=ax, sharey=ax)
        plot_2d_resid(resid, resid_range, 
                      pop1_label=labels[0], pop2_label=labels[1])

        ax = pylab.subplot(4,3,sax+10)
        flatresid = numpy.compress(numpy.logical_not(resid.mask.flat), 
                                   resid.flat)
        ax.hist(flatresid, bins=20, normed=True)
        ax.set_yticks([])
    pylab.show()
