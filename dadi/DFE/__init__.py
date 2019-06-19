"""
Fitting Distributions of Fitness Effects to population genomic data, both single-population and joint.
"""
from dadi.DFE.Cache1D_mod import Cache1D
from dadi.DFE.Cache2D_mod import Cache2D, mixture, mixture_symmetric_point_pos
from dadi.DFE import PDFs, DemogSelModels, Plotting

print("If you publish with the dadi DFE code, please cite Gutenkunst et al. (2009) PLoS "
      "Genetics and Kim, Huber, and Lohmueller (2017) Genetics "
      "(https://doi.org/10.1534/genetics.116.197145).")