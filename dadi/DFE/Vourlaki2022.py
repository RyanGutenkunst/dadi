import numpy as np, scipy.integrate
import dadi
from dadi.DFE import *

def Vourlaki_mixture(params, ns, s1, s2, theta, pts):
    """
    Inference model from Vourlaki et al.

    params = alpha, beta, ppos_wild, gamma_pos, pchange, pchange_pos
    """
    #params = alpha, beta, gamma_pos, ppos_wild, pchange, pchange_pos
    alpha, beta, ppos_wild, gamma_pos, pchange, pchange_pos = params

    # We'll scale by theta at the end of the script, so set theta=1 here.
    # Case in which gamma is negative and equal in the two pops
    m5 = s1.integrate([alpha, beta], None, PDFs.gamma, 1, None)
    # Case in which gamma is negative and indepenent in the two pops
    m6 = s2.integrate([alpha, beta], None, PDFs.biv_ind_gamma, 1, None,
                      exterior_int=True)

    # Cases in which gamma is positive in both pops.
    # For simplicity, using a single gamma value, rather than exponential dist
    #  that was simulated.
    try:
        m2 = s2.spectra[s2.gammas == gamma_pos,
                        s2.gammas == gamma_pos][0]
        # Even if selection coefficient changed to a different positive value,
        #   we just model them as equal.
        m3 = m2
    except IndexError:
        raise IndexError('Failed to find requested gamma_pos={0:.4f} '
                         'in cached spectra. Was it included '
                         'in additional_gammas during '
                         'cache generation?'.format(gamma_pos))

    # Cases in which gamma is positive in one pop and negative in the other.
    weights = PDFs.gamma(-s2.neg_gammas, [alpha, beta])

    # Case in which pop1 is positive and pop2 is negative.
    pos_neg_spectra = np.squeeze(s2.spectra[s2.gammas==gamma_pos, :len(s2.neg_gammas)])
    m4 = np.trapz(weights[:,np.newaxis,np.newaxis]*pos_neg_spectra,
                       s2.neg_gammas, axis=0)

    # Case in which pop2 is positive and pop1 is negative.
    neg_pos_spectra = np.squeeze(s2.spectra[:len(s2.neg_gammas), s2.gammas==gamma_pos])
    m7 = np.trapz(weights[:,np.newaxis,np.newaxis]*neg_pos_spectra,
                       s2.neg_gammas, axis=0)

    # Contributions to m4 and m7 for gammas that aren't covered by our cache
    # Probability toward gamma=0 that is not covered by our cache
    weight_neu, err = scipy.integrate.quad(PDFs.gamma, 0, -s2.neg_gammas[-1], args=[alpha, beta])
    # Probability toward gamma=-inf that is not covered by our cache
    weight_del, err = scipy.integrate.quad(PDFs.gamma, -s2.neg_gammas[0], np.inf, args=[alpha, beta])

    # In both cases, use the most neutral or deleterious spectra simulated
    m4 += pos_neg_spectra[0]*weight_del
    m4 += pos_neg_spectra[-1]*weight_neu
    m7 += neg_pos_spectra[0]*weight_del
    m7 += neg_pos_spectra[-1]*weight_neu

    # Weights for various parts of distribution checked against Figure S0
    fs = m5*(1-ppos_wild)*(1-pchange) +\
        m6*(1-ppos_wild)*pchange*(1-pchange_pos) +\
        m7*(1-ppos_wild)*pchange*pchange_pos +\
        m2*ppos_wild*(1-pchange) +\
        m3*ppos_wild*pchange*pchange_pos +\
        m4*ppos_wild*pchange*(1-pchange_pos)

    return theta*fs

def visual_validation():
    """
    Simulations with a simple model and extreme parameter values, to check
    functionality and build intuition.
    """
    demo_params = [2,2,0.2,0] #nu1, nu2, T, m
    ns, pts_l = [10,10], [30,35,40]

    s1 = Cache1D(demo_params, ns, DemogSelModels.split_mig_sel_single_gamma, pts=pts_l, 
                 gamma_pts=30, gamma_bounds=(1e-4, 200), mp=True, additional_gammas=[10])
    s2 = Cache2D(demo_params, ns, DemogSelModels.split_mig_sel, pts=pts_l, 
                 gamma_pts=30, gamma_bounds=(1e-4, 200), mp=True, additional_gammas=[10])

    alpha, beta, gamma_pos = 0.2, 10, 10

    import matplotlib.pyplot as plt
    fig = plt.figure(10, figsize=(10,8))
    fig.clear()

    axw = fig.add_subplot(3,2,5)
    axd = fig.add_subplot(3,2,6)

    # alpha, beta, ppos_wild, gamma_pos, pchange, pchange_pos
    fs = Vourlaki_mixture([alpha, beta, 0, gamma_pos, 0, 0], None, s1, s2, 1.0, None)
    ax = fig.add_subplot(3,2,1); dadi.Plotting.plot_single_2d_sfs(fs, ax=ax)
    ax.set_title('ppos_wild=0.0, pchange=0.0, pchange_pos=0.0')
    ax.text(0.02,0.98,'A', fontsize='x-large', va='top', transform=ax.transAxes)
    fsw, fsd = fs.filter_pops(tokeep=[1]), fs.filter_pops(tokeep=[2])
    axw.semilogy(fsw, '-o', label='A'); axd.semilogy(fsd, '-o', label='A')

    # alpha, beta, ppos_wild, gamma_pos, pchange, pchange_pos
    fs = Vourlaki_mixture([alpha, beta, 1.0, gamma_pos, 0, 0], None, s1, s2, 1.0, None)
    ax = fig.add_subplot(3,2,2); dadi.Plotting.plot_single_2d_sfs(fs, ax=ax)
    ax.set_title('ppos_wild=1.0, pchange=0.0, pchange_pos=0.0')
    ax.text(0.02,0.98,'B', fontsize='x-large', va='top', transform=ax.transAxes)
    fsw, fsd = fs.filter_pops(tokeep=[1]), fs.filter_pops(tokeep=[2])
    axw.semilogy(fsw, '-o', label='B'); axd.semilogy(fsd, '-o', label='B')

    # alpha, beta, ppos_wild, gamma_pos, pchange, pchange_pos
    fs = Vourlaki_mixture([alpha, beta, 0.0, gamma_pos, 1.0, 0], None, s1, s2, 1.0, None)
    ax = fig.add_subplot(3,2,3); dadi.Plotting.plot_single_2d_sfs(fs, ax=ax)
    ax.set_title('ppos_wild=0.0, pchange=1.0, pchange_pos=0.0')
    ax.text(0.02,0.98,'C', fontsize='x-large', va='top', transform=ax.transAxes)
    fsw, fsd = fs.filter_pops(tokeep=[1]), fs.filter_pops(tokeep=[2])
    axw.semilogy(fsw, '-o', label='C'); axd.semilogy(fsd, '-o', label='C')

    # alpha, beta, ppos_wild, gamma_pos, pchange, pchange_pos
    # This makes me think I got m4, m7 backwards
    fs = Vourlaki_mixture([alpha, beta, 0.0, gamma_pos, 1.0, 0.9], None, s1, s2, 1.0, None)
    ax = fig.add_subplot(3,2,4); dadi.Plotting.plot_single_2d_sfs(fs, ax=ax)
    ax.set_title('ppos_wild=0.0, pchange=1.0, pchange_pos=0.9')
    ax.text(0.02,0.98,'D', fontsize='x-large', va='top', transform=ax.transAxes)
    fsw, fsd = fs.filter_pops(tokeep=[1]), fs.filter_pops(tokeep=[2])
    axw.semilogy(fsw, '-o', label='D'); axd.semilogy(fsd, '-o', label='D')

    axw.set_title('wild'); axd.set_title('domesticate')
    axw.legend(); axd.legend()

    fig.tight_layout()
    fig.savefig('validation.pdf')
    plt.show()

if __name__ == "__main__":
    visual_validation()
