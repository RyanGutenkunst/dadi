"""
Routines for Plotting comparisons between model and data.

These can serve as inspiration for custom routines for one's own purposes.
Note that all the plotting is done with pylab. To see additional pylab methods:
"import pylab; help(pylab)". Pylab's many functions are documented at 
http://matplotlib.sourceforge.net/contents.html
"""

import matplotlib
import pylab
import numpy

#: Custom ticks that label only the lowest and highest bins in an FS plot.
class _sfsTickLocator(matplotlib.ticker.Locator):
    def __call__(self):
        'Return the locations of the ticks'

        try:
            vmin, vmax = self.axis.get_view_interval()
            dmin, dmax = self.axis.get_data_interval()
        except AttributeError:
            self.verify_intervals()
            vmin, vmax = self.viewInterval.get_bounds()
            dmin, dmax = self.dataInterval.get_bounds()

        tmin = max(vmin, dmin)
        tmax = min(vmax, dmax)

        return numpy.array([round(tmin)+0.5, round(tmax)-0.5])
#: Custom tick formatter
_ctf = matplotlib.ticker.FuncFormatter(lambda x,pos: '%i' % (x-0.4))


from dadi import Numerics, Inference

def plot_1d_comp_multinom(model, data, fig_num=None, residual='Anscombe',
                          plot_masked=False):
    """
    Mulitnomial comparison between 1d model and data.


    model: 1-dimensional model SFS
    data: 1-dimensional data SFS
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.
    residual: 'Anscombe' for Anscombe residuals, which are more normally
              distributed for Poisson sampling. 'linear' for the linear
              residuals, which can be less biased.
    plot_masked: Additionally plots (in open circles) results for points in the 
                 model or data that were masked.

    This comparison is multinomial in that it rescales the model to optimally
    fit the data.
    """
    masked_model, masked_data = Numerics.intersect_masks(model, data)
    masked_model = Inference.optimally_scaled_sfs(masked_model, masked_data)
    plot_1d_comp_Poisson(masked_model, masked_data, fig_num, residual,
                         plot_masked)

def plot_1d_comp_Poisson(model, data, fig_num=None, residual='Anscombe',
                         plot_masked=False):
    """
    Poisson comparison between 1d model and data.


    model: 1-dimensional model SFS
    data: 1-dimensional data SFS
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.
    residual: 'Anscombe' for Anscombe residuals, which are more normally
              distributed for Poisson sampling. 'linear' for the linear
              residuals, which can be less biased.
    plot_masked: Additionally plots (in open circles) results for points in the 
                 model or data that were masked.
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

    if plot_masked:
        pylab.semilogy(masked_data.data, '--ob', mfc='w', zorder=-100)
        pylab.semilogy(masked_model.data, '--or', mfc='w', zorder=-100)

    pylab.subplot(2,1,2, sharex = ax)
    if residual == 'Anscombe':
        resid = Inference.Anscombe_Poisson_residual(masked_model, masked_data)
    elif residual == 'linear':
        resid = Inference.linear_Poisson_residual(masked_model, masked_data)
    else:
        raise ValueError("Unknown class of residual '%s'." % residual)
    pylab.plot(resid, '-og')
    if plot_masked:
        pylab.plot(resid.data, '--og', mfc='w', zorder=-100)

    ax.set_xlim(0, data.shape[0]-1)
    pylab.show()

def plot_single_2d_sfs(sfs, vmin=None, vmax=None, ax=None, 
                       pop1_label= 'pop1', pop2_label='pop2',
                       extend='neither', colorbar=True):
    """
    Logarithmic heatmap of single 2d SFS.

    sfs: SFS to plot
    vmin: Values in sfs below vmin are masked in plot.
    vmax: Values in sfs above vmin saturate the color spectrum.
    ax: Axes object to plot into. If None, the result of pylab.gca() is used.
    pop1_label: Label for population 1.
    pop2_label: Label for population 2.
    extend: Whether the colorbar should have 'extension' arrows. See
            help(pylab.colorbar) for more details.
    colorbar: Should we plot a colorbar?
    """
    if ax is None:
        ax = pylab.gca()

    if vmin is None:
        vmin = sfs.min()
    if vmax is None:
        vmax = sfs.max()

    pylab.cm.hsv.set_under('w')
    mappable=ax.pcolor(numpy.ma.masked_where(sfs<vmin, sfs), 
                       cmap=pylab.cm.hsv, shading='flat',
                       norm = matplotlib.colors.LogNorm(vmin=vmin*(1-1e-3),
                                                        vmax=vmax))
    #cbticks = [numpy.round(vmin+0.05,1), numpy.round(vmax-0.05, 1)]

    # This can be passed to colorbar (format=format) to make ticks be 10^blah.
    # But I don't think it looks particularly nice.
    format = matplotlib.ticker.FormatStrFormatter('$10^{%.1f}$')
    ax.figure.colorbar(mappable, extend=extend)
    if not colorbar:
        del ax.figure.axes[-1]

    ax.plot([0,sfs.shape[1]],[0, sfs.shape[0]], '-k', lw=0.2)

    ax.set_ylabel(pop1_label, horizontalalignment='left')
    ax.set_xlabel(pop2_label, verticalalignment='bottom')

    ax.xaxis.set_major_formatter(_ctf)
    ax.xaxis.set_major_locator(_sfsTickLocator())
    ax.yaxis.set_major_formatter(_ctf)
    ax.yaxis.set_major_locator(_sfsTickLocator())
    for tick in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
        tick.set_visible(False)

    ax.set_xlim(0, sfs.shape[1])
    ax.set_ylim(0, sfs.shape[0])


def plot_2d_resid(resid, resid_range=None, ax=None, 
                  pop1_label = 'pop1', pop2_label='pop2', extend='neither',
                  colorbar=True):
    """
    Linear heatmap of 2d residual array.

    sfs: Residual array to plot.
    resid_range: Values > resid range or < resid_range saturate the color
                 spectrum.
    ax: Axes object to plot into. If None, the result of pylab.gca() is used.
    pop1_label: Label for population 1.
    pop2_label: Label for population 2.
    extend: Whether the colorbar should have 'extension' arrows. See
            help(pylab.colorbar) for more details.
    colorbar: Should we plot a colorbar?
    """
    if ax is None:
        ax = pylab.gca()

    if resid_range is None:
        resid_range = abs(resid).max()

    mappable=ax.pcolor(resid, cmap=pylab.cm.RdBu_r, vmin=-resid_range, 
                       vmax=resid_range, shading='flat')

    cbticks = [-resid_range, 0, resid_range]
    format = matplotlib.ticker.FormatStrFormatter('%.2g')
    ax.figure.colorbar(mappable, ticks=cbticks, format=format,
                       extend=extend)
    if not colorbar:
        del ax.figure.axes[-1]

    ax.plot([0,resid.shape[1]],[0, resid.shape[0]], '-k', lw=0.2)

    ax.set_ylabel(pop1_label, horizontalalignment='left')
    ax.set_xlabel(pop2_label, verticalalignment='bottom')

    ax.xaxis.set_major_formatter(_ctf)
    ax.xaxis.set_major_locator(_sfsTickLocator())
    ax.yaxis.set_major_formatter(_ctf)
    ax.yaxis.set_major_locator(_sfsTickLocator())
    for tick in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
        tick.set_visible(False)

    ax.set_xlim(0, resid.shape[1])
    ax.set_ylim(0, resid.shape[0])

# Used to determine whether colorbars should have 'extended' arrows
_extend_mapping = {(True, True): 'neither',
                   (False, True): 'min',
                   (True, False): 'max',
                   (False, False): 'both'}

def plot_2d_comp_multinom(model, data, vmin=None, vmax=None,
                          resid_range=None, fig_num=None,
                          pop_labels=['pop1', 'pop2'], residual='Anscombe',
                          adjust=True):
    """
    Mulitnomial comparison between 2d model and data.


    model: 2-dimensional model SFS
    data: 2-dimensional data SFS
    vmin, vmax: Minimum and maximum values plotted for sfs are vmin and
                vmax respectively.
    resid_range: Residual plot saturates at +- resid_range.
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.
    pop_labels: List of labels for populations 1 and 2.
    residual: 'Anscombe' for Anscombe residuals, which are more normally
              distributed for Poisson sampling. 'linear' for the linear
              residuals, which can be less biased.
    adjust: Should method use automatic 'subplots_adjust'? For advanced
            manipulation of plots, it may be useful to make this False.

    This comparison is multinomial in that it rescales the model to optimally
    fit the data.
    """
    masked_model, masked_data = Numerics.intersect_masks(model, data)
    masked_model = Inference.optimally_scaled_sfs(masked_model, masked_data)

    plot_2d_comp_Poisson(masked_model, masked_data, vmin=vmin, vmax=vmax,
                         resid_range=resid_range, fig_num=fig_num,
                         pop_labels=pop_labels, residual=residual,
                         adjust=adjust)
    
def plot_2d_comp_Poisson(model, data, vmin=None, vmax=None,
                         resid_range=None, fig_num=None,
                         pop_labels=['pop1', 'pop2'], residual='Anscombe',
                         adjust=True):
    """
    Poisson comparison between 2d model and data.


    model: 2-dimensional model SFS
    data: 2-dimensional data SFS
    vmin, vmax: Minimum and maximum values plotted for sfs are vmin and
                vmax respectively.
    resid_range: Residual plot saturates at +- resid_range.
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.
    pop_labels: List of labels for populations 1 and 2.
    residual: 'Anscombe' for Anscombe residuals, which are more normally
              distributed for Poisson sampling. 'linear' for the linear
              residuals, which can be less biased.
    adjust: Should method use automatic 'subplots_adjust'? For advanced
            manipulation of plots, it may be useful to make this False.
    """
    masked_model, masked_data = Numerics.intersect_masks(model, data)

    if fig_num is None:
        f = pylab.gcf()
    else:
        f = pylab.figure(fig_num, figsize=(7,7))

    pylab.clf()
    if adjust:
        pylab.subplots_adjust(bottom=0.07, left=0.07, top=0.95, right=0.95)

    max_toplot = max(masked_model.max(), masked_data.max())
    min_toplot = min(masked_model.min(), masked_data.min())
    if vmax is None:
        vmax = max_toplot
    if vmin is None:
        vmin = min_toplot
    extend = _extend_mapping[vmin <= min_toplot, vmax >= max_toplot]

    ax = pylab.subplot(2,2,1)
    plot_single_2d_sfs(masked_data, vmin=vmin, vmax=vmax,
                       pop1_label=pop_labels[0], pop2_label=pop_labels[1],
                       colorbar=False)

    pylab.subplot(2,2,2, sharex=ax, sharey=ax)
    plot_single_2d_sfs(masked_model, vmin=vmin, vmax=vmax,
                       pop1_label=pop_labels[0], pop2_label=pop_labels[1],
                       extend=extend )

    if residual == 'Anscombe':
        resid = Inference.Anscombe_Poisson_residual(masked_model, masked_data,
                                              mask=vmin)
    elif residual == 'linear':
        resid = Inference.linear_Poisson_residual(masked_model, masked_data,
                                            mask=vmin)
    else:
        raise ValueError("Unknown class of residual '%s'." % residual)

    if resid_range is None:
        resid_range = max((abs(resid.max()), abs(resid.min())))
    resid_extend = _extend_mapping[-resid_range <= resid.min(), 
                                   resid_range >= resid.max()]

    pylab.subplot(2,2,3, sharex=ax, sharey=ax)
    plot_2d_resid(resid, resid_range, 
                  pop1_label=pop_labels[0], pop2_label=pop_labels[1],
                  extend=resid_extend)

    ax = pylab.subplot(2,2,4)
    flatresid = numpy.compress(numpy.logical_not(resid.mask.ravel()), 
                               resid.ravel())
    ax.hist(flatresid, bins=20, normed=True)
    ax.set_yticks([])
    pylab.show()

def plot_3d_comp_multinom(model, data, vmin=None, vmax=None,
                          resid_range=None, fig_num=None,
                          pop_labels=['pop1', 'pop2', 'pop3'], 
                          residual='Anscombe', adjust=True):
    """
    Multinomial comparison between 3d model and data.


    model: 3-dimensional model SFS
    data: 3-dimensional data SFS
    vmin, vmax: Minimum and maximum values plotted for sfs are vmin and
                vmax respectively.
    resid_range: Residual plot saturates at +- resid_range.
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.
    pop_labels: List of labels for populations 1, 2, and 3.
    residual: 'Anscombe' for Anscombe residuals, which are more normally
              distributed for Poisson sampling. 'linear' for the linear
              residuals, which can be less biased.
    adjust: Should method use automatic 'subplots_adjust'? For advanced
            manipulation of plots, it may be useful to make this False.

    This comparison is multinomial in that it rescales the model to optimally
    fit the data.
    """
    masked_model, masked_data = Numerics.intersect_masks(model, data)
    masked_model = Inference.optimally_scaled_sfs(masked_model, masked_data)

    plot_3d_comp_Poisson(masked_model, masked_data, vmin=vmin, vmax=vmax,
                         resid_range=resid_range, fig_num=fig_num,
                         pop_labels=pop_labels, residual=residual,
                         adjust=adjust)

def plot_3d_comp_Poisson(model, data, vmin=None, vmax=None,
                         resid_range=None, fig_num=None,
                         pop_labels=['pop1', 'pop2', 'pop3'], 
                         residual='Anscombe', adjust=True):
    """
    Poisson comparison between 3d model and data.


    model: 3-dimensional model SFS
    data: 3-dimensional data SFS
    vmin, vmax: Minimum and maximum values plotted for sfs are vmin and
                vmax respectively.
    resid_range: Residual plot saturates at +- resid_range.
    fig_num: Clear and use figure fig_num for display. If None, an new figure
             window is created.
    pop_labels: List of labels for populations 1, 2, and 3.
    residual: 'Anscombe' for Anscombe residuals, which are more normally
              distributed for Poisson sampling. 'linear' for the linear
              residuals, which can be less biased.
    adjust: Should method use automatic 'subplots_adjust'? For advanced
            manipulation of plots, it may be useful to make this False.
    """
    masked_model, masked_data = Numerics.intersect_masks(model, data)

    if fig_num is None:
        f = pylab.gcf()
    else:
        f = pylab.figure(fig_num, figsize=(8,10))

    pylab.clf()
    if adjust:
        pylab.subplots_adjust(bottom=0.07, left=0.07, top=0.95, right=0.95)

    modelmax = max(masked_model.sum(axis=sax).max() for sax in range(3))
    datamax = max(masked_data.sum(axis=sax).max() for sax in range(3))
    modelmin = min(masked_model.sum(axis=sax).min() for sax in range(3))
    datamin = min(masked_data.sum(axis=sax).min() for sax in range(3))
    max_toplot = max(modelmax, datamax)
    min_toplot = min(modelmin, datamin)

    if vmax is None:
        vmax = max_toplot
    if vmin is None:
        vmin = min_toplot
    extend = _extend_mapping[vmin <= min_toplot, vmax >= max_toplot]

    # Calculate the residuals
    if residual == 'Anscombe':
        resids = [Inference.\
                  Anscombe_Poisson_residual(masked_model.sum(axis=2-sax), 
                                            masked_data.sum(axis=2-sax), 
                                            mask=vmin) for sax in range(3)]
    elif residual == 'linear':
        resids =[Inference.\
                 linear_Poisson_residual(masked_model.sum(axis=2-sax), 
                                         masked_data.sum(axis=2-sax), 
                                         mask=vmin) for sax in range(3)]
    else:
        raise ValueError("Unknown class of residual '%s'." % residual)


    min_resid = min([r.min() for r in resids])
    max_resid = max([r.max() for r in resids])
    if resid_range is None:
        resid_range = max((abs(max_resid), abs(min_resid)))
    resid_extend = _extend_mapping[-resid_range <= min_resid, 
                                   resid_range >= max_resid]

    for sax in range(3):
        marg_data = masked_data.sum(axis=2-sax)
        marg_model = masked_model.sum(axis=2-sax)

        labels = list(pop_labels[:])
        del labels[2-sax]

        ax = pylab.subplot(4,3,sax+1)
        plot_colorbar = (sax == 2)
        plot_single_2d_sfs(marg_data, vmin=vmin, vmax=vmax,
                           pop1_label=labels[0], pop2_label=labels[1],
                           extend=extend, colorbar=plot_colorbar)

        pylab.subplot(4,3,sax+4, sharex=ax, sharey=ax)
        plot_single_2d_sfs(marg_model, vmin=vmin, vmax=vmax,
                           pop1_label=labels[0], pop2_label=labels[1],
                           extend=extend, colorbar=False)

        resid = resids[sax]
        pylab.subplot(4,3,sax+7, sharex=ax, sharey=ax)
        plot_2d_resid(resid, resid_range, 
                      pop1_label=labels[0], pop2_label=labels[1],
                      extend=resid_extend, colorbar=plot_colorbar)

        ax = pylab.subplot(4,3,sax+10)
        flatresid = numpy.compress(numpy.logical_not(resid.mask.ravel()), 
                                   resid.ravel())
        ax.hist(flatresid, bins=20, normed=True)
        ax.set_yticks([])
    pylab.show()
