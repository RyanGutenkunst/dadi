#!/bin/env python

import socket, sys
import numpy as np
import dadi
import models

import argparse
parser = argparse.ArgumentParser(description='Benchmark dadi integration')
parser.add_argument('--cuda', action='store_true', help='run using GPU')
parser.add_argument('--RAM', type=float, default=0.01, help='approximate amount of RAM to use (in GB)')
args = parser.parse_args()

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

# Burn in to ensure GPU kernels are loaded
for ii in range(3):
    models.OutOfAfrica_2L06(10, variant='original')
    models.OutOfAfrica_2L06(10, variant='const_int')

maxpts = int(dadi.RAM_to_pts(args.RAM, 2))
pts_l_2D = 10 * 2**np.arange(0,np.log2(maxpts//10), dtype=int)

print('# Model: 2D Original')
for pts in pts_l_2D:
    _,t = models.OutOfAfrica_2L06_timed(pts, variant='original')
    print('pts: {0}, time: {1:.4g}'.format(pts, t), flush=True)
print()

print('# Model: 2D Cached integration')
for pts in pts_l_2D[:-1]:
    _,t = models.OutOfAfrica_2L06_timed(pts, variant='cached_int')
    print('pts: {0}, time: {1:.4g}'.format(pts, t), flush=True)
print()

maxpts = int(dadi.RAM_to_pts(args.RAM, 3))
pts_l_3D = 10 * 2**np.arange(0,np.log2(maxpts//10), dtype=int)

print('# Model: 3D Original')
for pts in pts_l_3D:
    _, t = models.OutOfAfrica_3G09_timed(pts, variant='original')
    print('pts: {0}, time: {1:.4g}'.format(pts, t), flush=True)
print()

print('# Model: 3D Constant')
for pts in pts_l_3D:
    _, t = models.OutOfAfrica_3G09_timed(pts, variant='const')
    print('pts: {0}, time: {1:.4g}'.format(pts, t), flush=True)
print()

print('# Model: 3D Cached integration')
for pts in pts_l_3D[:-1]:
    _, t = models.OutOfAfrica_3G09_timed(pts, variant='cached_int')
    print('pts: {0}, time: {1:.4g}'.format(pts, t), flush=True)
print()

maxpts = int(dadi.RAM_to_pts(args.RAM, 4))
pts_l_4D = 10 * 2**np.arange(0,np.log2(maxpts//10), dtype=int)
print('# Model: 4D Original')
for pts in pts_l_4D:
    _,t = models.NewWorld_4G09_timed(pts, variant='original')
    print('pts: {0}, time: {1:.4g}'.format(pts, t), flush=True)
print()

maxpts = int(dadi.RAM_to_pts(args.RAM, 5))
pts_l_5D = 10 * 2**np.arange(0,np.log2(maxpts//10), dtype=int)
print('# Model: 5D Original')
for pts in pts_l_5D:
    _, t = models.OutOfAfricaArchaicAdmixture_5R19_timed(pts, variant='original')
    print('pts: {0}, time: {1:.4g}'.format(pts, t), flush=True)
print()