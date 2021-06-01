import dadi

def OutOfAfrica(params, ns, pts):
    nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs, mAfB, mAfEu, mAfAs, mEuAs, TAf, TB, TEuAs = params
    xx = dadi.Numerics.default_grid(pts)

    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.Integration.one_pop(phi, xx, TAf, nu = nuAf)

    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.Integration.two_pops(phi, xx, TB, nu1 = nuAf, nu2 = nuB, m12 = mAfB, m21 = mAfB)

    phi = dadi.PhiManip.phi_2D_to_3D_split_2(xx, phi)

    nuEu_func = lambda t : nuEu0 * (nuEu/nuEu0) ** (t/TEuAs)
    nuAs_func = lambda t : nuAs0 * (nuAs/nuAs0) ** (t/TEuAs)
    phi = dadi.Integration.three_pops(phi, xx, TEuAs, nu1 = nuAf, nu2 = nuEu_func, nu3 = nuAs_func, 
        m12 = mAfEu, m13 = mAfAs, m21 = mAfEu, m23 = mEuAs, m31 = mAfAs, m32 = mEuAs)

    fs = dadi.Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs


def browning_america(params, ns, pts):
    nuAFR, nuOOA, \
    nuEUR0, nuEUR, nuEUR_F, \
    nuEAS0, nuEAS, nuEAS_F,\
    nuAdmix0, nuAdmixF,\
    mAFR_OOA, mAFR_EUR, mAFR_EAS, mEUR_EAS, \
    TAFR, TOOA, TEUR_EAS, TAdmix, prop = params

    xx = dadi.Numerics.default_grid(pts)

    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.Integration.one_pop(phi, xx, TAFR, nu = nuAFR)

    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.Integration.two_pops(phi, xx, TOOA, nu1 = nuAFR, nu2 = nuOOA, m12 = mAFR_OOA, m21 = mAFR_OOA)

    phi = dadi.PhiManip.phi_2D_to_3D_split_2(xx, phi)

    nuEUR_func = lambda t : nuEUR0 * (nuEUR/nuEUR0) ** (t/TEUR_EAS)
    nuEAS_func = lambda t : nuEAS0 * (nuEAS/nuEAS0) ** (t/TEUR_EAS)
    phi = dadi.Integration.three_pops(phi, xx, TEUR_EAS, nu1 = nuAFR, nu2 = nuEUR_func, nu3 = nuEAS_func, 
        m12 = mAFR_EUR, m13 = mAFR_EAS, m21 = mAFR_EUR, m23 = mEUR_EAS, m31 = mAFR_EAS, m32 = mEUR_EAS)

    phi = dadi.PhiManip.phi_3D_to_4D(phi, prop[0],prop[1], xx,xx,xx,xx)

    nuEUR_F_func = lambda t : nuEUR * (nuEUR_F/nuEUR) ** (t/TAdmix)
    nuEAS_F_func = lambda t : nuEAS * (nuEAS_F/nuEAS) ** (t/TAdmix)
    nuAdmixF_func = lambda t : nuAdmix0 * (nuAdmixF/nuAdmix0) ** (t/TAdmix)

    phi = dadi.Integration.four_pops(phi, xx, TAdmix, nu1 = nuAFR, 
        nu2 = nuEUR_F_func, nu3 = nuEAS_F_func,  nu4 = nuAdmixF_func,
        m12 = mAFR_EUR, m13 = mAFR_EAS, m14 = 0,
        m21 = mAFR_EUR, m23 = mEUR_EAS, m24 = 0,
        m31 = mAFR_EAS, m32 = mEUR_EAS, m34= 0,
        m41 = 0, m42 = 0, m43 = 0)

    fs = dadi.Spectrum.from_phi(phi, ns, (xx, xx, xx, xx))
    return fs













