"""
An example of obtaining the sample triallelic frequency spectrum for a simple two epoch demography, with selection
"""

import dadi
import numpy as np, scipy, matplotlib

nu, T = 2.0, 0.1 # instantaneous population size change (doubled in size) 0.1 time units (in 2Ne generations) ago
sig1 = -1.0
sig2 = 0.0
theta1 = 1.
theta2 = 1.
misid = 0.0 # no misidentification
dt = 0.001 # time step of integration - should automate?

params = [nu,T,sig1,sig2,theta1,theta2,misid,dt]

grid_pts = [40,60,80] # evaluate over these grid points, then extrapolate to $\Delta = 0$
ns = 12 # number of observed samples

fs0 = dadi.Triallele.demographics.two_epoch(params,ns,grid_pts[0], folded = False, misid = False)
fs1 = dadi.Triallele.demographics.two_epoch(params,ns,grid_pts[1], folded = False, misid = False)
fs2 = dadi.Triallele.demographics.two_epoch(params,ns,grid_pts[2], folded = False, misid = False)

fs = dadi.Numerics.quadratic_extrap((fs0,fs1,fs2),(fs0.extrap_x,fs1.extrap_x,fs2.extrap_x))

