


#simple model to test splits up to three populations
model = "gutenkunst_ooa.yml"
sampled_demes = ["YRI", "CEU", "CHB"]
sample_sizes = [10, 10, 10]

#simple model to test admix using all populations to form a new one
model = "browning_america.yml"
sampled_demes = ['AFR', 'EUR', 'EAS', 'ADMIX']
sample_sizes = [10, 10, 10, 10]
# sampled_demes = ['ADMIX']
# sample_sizes = [10]

#simple model to test branches and pulses
model = "offshoots.yml"
sampled_demes=['ancestral', 'offshoot1', 'offshoot2']
sample_sizes = [10, 10, 10]

pts_l = [20,30,40]

import dadi
fs_dadi = dadi.Spectrum.from_demes(model, sampled_demes=sampled_demes, sample_sizes=sample_sizes, pts=pts_l)
print(fs_dadi.S())

import moments
fs_moments = moments.Spectrum.from_demes(model, sampled_demes=sampled_demes, sample_sizes=sample_sizes)
print(fs_moments.S())

if 'gutenkunst_ooa' in model:
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

    params = [1.6849315068493151, 0.2876712328767123, 
    0.129, 3.74, 0.07, 7.29, 
    3.65, 0.438, 0.2774, 1.4016,
    0.2191780821917808, 0.3254794520547945, 0.05808219178082192]

    func_ex_OutOfAfrica = dadi.Numerics.make_extrap_func(OutOfAfrica)
    fs_OOA = func_ex_OutOfAfrica(params, sample_sizes, pts_l)
    print(fs_OOA.S())

    print()









