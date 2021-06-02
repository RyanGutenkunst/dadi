


# #simple model to test splits up to three populations
# model = "gutenkunst_ooa.yml"
# sampled_demes = ["YRI", "CEU", "CHB"]
# sample_sizes = [10, 10, 10]

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
fs_demes = dadi.Spectrum.from_demes(model, sampled_demes=sampled_demes, sample_sizes=sample_sizes, pts=pts_l)
print(fs_demes.S())

import testing_models

if 'gutenkunst_ooa' in model:
    dadi_model = testing_models.OutOfAfrica

    params = [1.6849315068493151, 0.2876712328767123, 
    0.136986301369863, 4.071917808219178, 0.06986301369863014, 7.409589041095891, 
    3.65, 0.438, 0.2774, 1.4016,
    0.2191780821917808, 0.3254794520547945, 0.05808219178082192]

    func_ex = dadi.Numerics.make_extrap_func(dadi_model)
    fs_dadi = func_ex(params, sample_sizes, pts_l)
    print(fs_dadi.S())

if 'browning_america' in model:
    dadi_model = testing_models.browning_america

    params = [
    1.9800273597811218, 0.2545827633378933, #AFR and OOA size changes
    0.13679890560875513, 4.447102197124753, 4.6564979480164155, #EUR size changes
    0.06976744186046512, 5.915026764391873, 6.27250341997264, #EAS size changes
    4.1039671682626535, 7.477975376196991, #Admix size changes
    2.193, 0.3655, 0.114036, 0.454682, #Migration rates
	0.265389876880985, 0.07660738714090287, 0.06210670314637483, 0.0008207934336525308, #Ts
	[0.167, 0.333, 0.5] #proportions
	]

    func_ex = dadi.Numerics.make_extrap_func(dadi_model)
    fs_dadi = func_ex(params, sample_sizes, pts_l)
    print(fs_dadi.S())

if 'offshoots' in model:
    dadi_model = testing_models.offshoots

    params = [
    1.0, 0.2, 0.1, #size changes
    0.02, 0.04, 0.2, #Migration rates
    0.0, 0.25, 0.15, 0.05, 0.025, 0.25, #Ts
    0.1 #proportions
    ]

    func_ex = dadi.Numerics.make_extrap_func(dadi_model)
    fs_dadi = func_ex(params, sample_sizes, pts_l)
    print(fs_dadi.S())





