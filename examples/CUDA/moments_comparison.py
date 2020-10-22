#!/bin/env python

import socket, sys, time
import numpy as np

from models_moments import *

import argparse
parser = argparse.ArgumentParser(description='Benchmark dadi versus moments')
parser.add_argument('--cuda', action='store_true', help='run using dadi GPU')
parser.add_argument('--moments', action='store_true', help='run using moments')
parser.add_argument('--RAM', type=float, default=0.01, help='approximate amount of RAM to use (in GB)')
args = parser.parse_args()

def pts_l_func(n):
    return (int(n*1.1)+2, int(n*1.2)+4, int(n*1.3)+6)

if args.cuda:
    if not dadi.cuda_enabled(True):
        raise ValueError('Failed to initialize CUDA')

print('# Host: {0}'.format(socket.getfqdn()))
if args.moments:
    print('# Using moments')
else:
    print('# Using dadi')
print('# CUDA enabled: {0}'.format(dadi.cuda_enabled()))
if dadi.cuda_enabled():
    import pycuda.autoinit
    device = pycuda.autoinit.device
    print('# GPU: {0} ({1:.1f} GB RAM)'.format(device.name(), 
          device.total_memory()/1024**3))
print()

# Burn in to ensure GPU kernels are loaded
for ii in range(3):
    models.OutOfAfrica_2L06(10, variant='original')
    models.OutOfAfrica_2L06(10, variant='cached_int')

maxpts = int(dadi.RAM_to_pts(args.RAM, 2)/1.5)
n_l_2D = 10 * 2**np.arange(0,np.log2(maxpts//10), dtype=int)
print('# Model: 2D Original')
for n in n_l_2D:
    start = time.time()
    if args.moments:
        fs = OutOfAfrica_2L06_moments(n)
    else:
        fs = OutOfAfrica_2L06_dadi(n, pts_l_func(n))
    print('n: {0}, time: {1:.4g}'.format(n, time.time()-start), flush=True)
print()

maxpts = int(dadi.RAM_to_pts(args.RAM, 3)/1.5)
n_l_3D = 10 * 2**np.arange(0,np.log2(maxpts//10), dtype=int)
print('# Model: 3D Original')
for n in n_l_3D:
    start = time.time()
    if args.moments:
        fs = OutOfAfrica_3G09_moments(n)
    else:
        fs = OutOfAfrica_3G09_dadi(n, pts_l_func(n))
    print('n: {0}, time: {1:.4g}'.format(n, time.time()-start), flush=True)
print()

maxpts = int(dadi.RAM_to_pts(args.RAM, 4)/1.5)
n_l_4D = 10 * 2**np.arange(0,np.log2(maxpts//10), dtype=int)
print('# Model: 4D Original')
for n in n_l_4D:
    start = time.time()
    if args.moments:
        fs = NewWorld_4G09_noadmix_moments(n)
    else:
        fs = NewWorld_4G09_noadmix_dadi(n, pts_l_func(n))
    print('n: {0}, time: {1:.4g}'.format(n, time.time()-start), flush=True)
print()

maxpts = int(dadi.RAM_to_pts(args.RAM, 5)/1.5)
n_l_5D = 10 * 2**np.arange(0,np.log2(maxpts//10), dtype=int)
print('# Model: 5D Original')
for n in n_l_5D:
    start = time.time()
    if args.moments:
        fs = OutOfAfricaArchaicAdmixture_5R19_moments(n)
    else:
        fs = OutOfAfricaArchaicAdmixture_5R19_dadi(n, pts_l_func(n))
    print('n: {0}, time: {1:.4g}'.format(n, time.time()-start), flush=True)
print()
