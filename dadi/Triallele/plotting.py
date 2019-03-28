import numpy as np, dadi
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import newaxis as nuax
import matplotlib.image as mpimg
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from matplotlib import ticker

import matplotlib
# Set fontsize to 10
matplotlib.rc('font',**{'family':'sans-serif',
                        'sans-serif':['Helvetica'],
                        'style':'normal',
                        'size':10 })
# Set label tick sizes to 8
matplotlib.rc('xtick', labelsize=8)
matplotlib.rc('ytick', labelsize=8)

def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

_cmap = plt.get_cmap('cubehelix_r')
_new_cmap = _truncate_colormap(_cmap, 0.1, 1.0)
_new_cmap2 = _truncate_colormap(_cmap, 0.24, 1.0)

def _fold(spectrum):
    spectrum = dadi.Spectrum(spectrum)
    if spectrum.mask[1,2] == True:
        print("error: trying to fold a spectrum that is already folded")
        return spectrum
    else:
        spectrum = (spectrum + np.transpose(spectrum))
        for ii in range(len(spectrum)):
            spectrum[ii,ii] = spectrum[ii,ii]/2
        spectrum.mask[0,:] = True
        spectrum.mask[:,0] = True
        for ii in range(len(spectrum)):
            spectrum.mask[ii,ii+1:] = True
            spectrum.mask[ii,len(spectrum)-1-ii:] = True
        return spectrum

def plot_single_trispectrum(sfs, folded=False, cmap=_new_cmap, vmin=None, vmax=None, colorbar=False, fraction=.046):
    """
    Plots a single triallelic spectrum (sfs)
    ~~~~8/13: to check - does it need to be given an unfolded spectrum? what if we want
    ~~~~ to plot a folded spectrum...
    folded: True if we want to fold the spectrum before plotting, False if to remain unfolded
    cmap: Define the colormap to use. Default is same colormap as used in figures from Ragsdale et al (2016)
    vmin: lower limit of colormap and smallest value to be plotted
    vmax: upper limit of colormap
    colorbar: True to show colorbar for Count
    """
    ax = plt.gca()
    if vmin is None:
        vmin = sfs.min()
    if vmax is None:
        vmax = sfs.max()
        
    if folded == True:
        sfs = _fold(sfs)
        mappable = ax.pcolor(np.transpose(np.ma.masked_where(sfs < vmin, sfs)), 
                    norm=LogNorm(vmin=vmin, vmax=vmax), 
                    vmin = vmin, vmax = vmax, cmap=_cmap)
    else:
        mappable = ax.pcolor(np.ma.masked_where(sfs < vmin, sfs), 
                    norm=LogNorm(vmin=vmin, vmax=vmax), 
                    vmin = vmin, vmax = vmax, cmap=_cmap)
    
    ax.set_xlim(0, sfs.shape[1])
    if folded == True:
        ax.set_ylim(0, sfs.shape[0]/2)
        ax.set_xlabel('Major derived allele frequency')
        ax.set_ylabel('Minor derived allele frequency')
    else:
        ax.set_ylim(0, sfs.shape[0])
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Frequency')
    
    if colorbar == True:
        cbar = ax.figure.colorbar(mappable,fraction=fraction, pad=0.04,
                                  shrink=0.85)
        #cbar.ax.tick_params(labelsize=6)
        cbar.set_label(label='Count', fontsize=10)
        return cbar
        

def plot_trispectrum_comp(sfs1, sfs2, folded=False, cmap=_new_cmap, vmin=None, vmax=None, resid_range=None, colorbar=False, title1="sfs1", title2="sfs2", title3="Residual", fraction=.046):
    """
    Plots the two spectra and the residual ( (sfs1-sfs2)/sqrt(sfs1) )
    folded: True if we want to fold the spectrum before plotting, False if to remain unfolded
    cmap: Define the colormap to use. Default is same colormap as used in figures from Ragsdale et al (2016)
    vmin: lower limit of colormap and smallest value to be plotted for frequency spectra
    vmax: upper limit of colormap for frequency spectra
    resid_range: residual colormap ranges from -resid_range to +resid_range
    colorbar: True to show colorbars for Count and Residual
    """
    fig = plt.figure(np.random.randint(1000))

    if vmin is None:
        vmin = np.min((sfs1.min(),sfs2.min()))
    if vmax is None:
        vmax = np.max((sfs1.max(),sfs2.max()))
    
    ax1 = plt.subplot(3,1,1)
    plot_single_trispectrum(sfs1, folded=folded, cmap=cmap, vmin=vmin, vmax=vmax, colorbar=colorbar, fraction=.046)
    ax1.set_title(title1)
    
    ax2 = plt.subplot(3,1,2)
    plot_single_trispectrum(sfs2, folded=folded, cmap=cmap, vmin=vmin, vmax=vmax, colorbar=colorbar, fraction=.046)
    ax2.set_title(title2)
    
    ax3 = plt.subplot(3,1,3)
    ax3.set_title(title3)
    
    residuals = (sfs1 - sfs2)/np.sqrt(sfs1)
    if folded == True:
        residuals = _fold(residuals)
        if resid_range == None:
            mappable = ax3.pcolor(np.transpose(residuals), vmin = -np.max(abs(residuals)), vmax = np.max(abs(residuals)), cmap='RdBu')
        else:
            mappable = ax3.pcolor(np.transpose(residuals), vmin = -resid_range, vmax = resid_range, cmap='RdBu')
    else:
        if resid_range == None:
            mappable = ax3.pcolor(residuals, vmin = -np.max(abs(residuals)), vmax = np.max(abs(residuals)), cmap='RdBu')
        else:
            mappable = ax3.pcolor(residuals, vmin = -resid_range, vmax = resid_range, cmap='RdBu')
    
    ax3.set_xlim(0, sfs1.shape[1])
    if folded == True:
        ax3.set_ylim(0, sfs1.shape[0]/2)
    else:
        ax3.set_ylim(0, sfs1.shape[0])
    
    if colorbar == True:
        cbar = ax3.figure.colorbar(mappable,fraction=fraction, pad=0.04,
                                  shrink=0.85)
        #cbar.ax.tick_params(labelsize=6)
        cbar.set_label(label='Residual', fontsize=10)
        return cbar
    
    