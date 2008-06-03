import pylab

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
