#!/bin/env python

import socket, sys
import numpy as np
import dadi
import models

import argparse
parser = argparse.ArgumentParser(description='Benchmark dadi integration')
parser.add_argument('--cuda', action='store_true', help='run using GPU')
parser.add_argument('--moments', action='store_true', help='run using moments')
parser.add_argument('--RAM', type=float, default=0.01, help='approximate amount of RAM to use (in GB)')
args = parser.parse_args()

if args.moments:
    print('# Using moments')
    import models_moments

def pts_l_func(n):
    """
    Plausible pts_l for sample size n
    """
    return (int(n*1.1)+2, int(n*1.2)+4, int(n*1.3)+6)

if args.cuda:
    if not dadi.cuda_enabled(True):
        raise ValueError('Failed to initialize CUDA')

print('# Host: {0}'.format(socket.getfqdn()))
print('# CUDA enabled: {0}'.format(dadi.cuda_enabled()))
if dadi.cuda_enabled():
    import pycuda.autoinit
    device = pycuda.autoinit.device
    print('# GPU: {0} ({1:.1f} GB RAM)'.format(device.name(), 
          device.total_memory()/1024**3))
print()

# Extra factor of 1.3 to account for additional memory used
# in pts_l extrapolation.
maxpts = int(dadi.RAM_to_pts(args.RAM, 2)/1.3)
n_l_2D = 10 * 2**np.arange(0,np.log2(maxpts//10), dtype=int)
print('# Model: 2D')
for n in n_l_2D:
    # In each case, we run the calculation twice, so any caches can be filled.
    # We do this because the most relevant scenario is repeated model evaluation
    #  during optimization
    if args.moments:
        _,t = models_moments.OutOfAfrica_2L06_moments_timed(n)
        _,t = models_moments.OutOfAfrica_2L06_moments_timed(n)
    else:
        _, t = models.OutOfAfrica_2L06_timed(n, pts_l_func(n))
        _, t = models.OutOfAfrica_2L06_timed(n, pts_l_func(n))
    print('n: {0}, time: {1:.4g}'.format(n, t), flush=True)
print()

maxpts = int(dadi.RAM_to_pts(args.RAM, 3)/1.3)
n_l_3D = 10 * 2**np.arange(0,np.log2(maxpts//10), dtype=int)
print('# Model: 3D')
for n in n_l_3D:
    if args.moments:
        _, t = models_moments.OutOfAfrica_3G09_moments_timed(n)
        _, t = models_moments.OutOfAfrica_3G09_moments_timed(n)
    else:
        _, t = models.OutOfAfrica_3G09_timed(n, pts_l_func(n))
        _, t = models.OutOfAfrica_3G09_timed(n, pts_l_func(n))
    print('n: {0}, time: {1:.4g}'.format(n, t), flush=True)
print()

maxpts = int(dadi.RAM_to_pts(args.RAM, 4)/1.3)
n_l_4D = 4 * 2**np.arange(0,np.log2(maxpts//4), dtype=int)
print('# Model: 4D')
for n in n_l_4D:
    if args.moments:
        _,t = models_moments.NewWorld_4G09_noadmix_moments_timed(n)
        _,t = models_moments.NewWorld_4G09_noadmix_moments_timed(n)
    else:
        _,t = models.NewWorld_4G09_timed(n, pts_l_func(n), variant='no_admixture')
        _,t = models.NewWorld_4G09_timed(n, pts_l_func(n), variant='no_admixture')
    print('n: {0}, time: {1:.4g}'.format(n, t), flush=True)
print()

maxpts = int(dadi.RAM_to_pts(args.RAM, 5)/1.3)
n_l_5D = 2 * 2**np.arange(1,np.log2(maxpts//2), dtype=int)
print('# Model: 5D')
for n in n_l_5D:
    if args.moments:
        _,t = models_moments.OutOfAfricaArchaicAdmixture_5R19_moments_timed(n)
        _,t = models_moments.OutOfAfricaArchaicAdmixture_5R19_moments_timed(n)
    else:
        _,t = models.OutOfAfricaArchaicAdmixture_5R19_timed(n, pts_l_func(n))
        _,t = models.OutOfAfricaArchaicAdmixture_5R19_timed(n, pts_l_func(n))
    print('n: {0}, time: {1:.4g}'.format(n, t), flush=True)
print()
