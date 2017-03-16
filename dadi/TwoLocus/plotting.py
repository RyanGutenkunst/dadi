import numpy as np
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d as mplot3d

# adapted from dadi's Plotting methods
def plot_3d(fs, vmin=None, vmax=None, max_sum=None, max_index=None, show=True, colorbar=False):
    """
    vmin - minimum value to plot
    vmax - maximum value to plot
    max_sum - slice spectrum on diagonal plane, plotting points closer to the origin than the plane
    max_index - only show entries this distance from one of the axes planes
    """
    fig = plt.figure()
    plt.clf()
    ax = mplot3d.Axes3D(fig)

    if vmin==None:
        vmin = np.min(fs[fs>0])
    if vmax==None:
        vmax = np.max(fs[fs>0])

    fs = np.swapaxes(fs,0,2)

    toplot = np.logical_not(fs.mask)
    toplot = np.logical_and(toplot, fs.data >= vmin)
    if max_sum != None:
        iis = np.where(toplot)[0]
        jjs = np.where(toplot)[1]
        kks = np.where(toplot)[2]
        for ll in range(len(iis)):
            ii = iis[ll]
            jj = jjs[ll]
            kk = kks[ll]
            if ii+jj+kk > max_sum:
                toplot[ii,jj,kk] = False
    if max_index != None:
        iis = np.where(toplot)[0]
        jjs = np.where(toplot)[1]
        kks = np.where(toplot)[2]
        for ll in range(len(iis)):
            ii = iis[ll]
            jj = jjs[ll]
            kk = kks[ll]
            if ii > max_index and jj > max_index and kk > max_index:
                toplot[ii,jj,kk] = False

    normalized = (np.log(fs)-np.log(vmin))\
            /(np.log(vmax)-np.log(vmin))
    normalized = np.minimum(normalized, 1)
    # scrunch by a factor 
    # XXX: this is really hacky
    factor = .1
    normalized = (1-2*factor)*normalized + .2
    colors = plt.cm.cubehelix_r(normalized)

    # We draw by calculating which faces are visible and including each as a
    # polygon.
    polys, polycolors = [],[]
    for ii in range(fs.shape[0]):
        for jj in range(fs.shape[1]):
            for kk in range(fs.shape[2]):
                if not toplot[ii,jj,kk]:
                    continue
                if kk < fs.shape[2]-1 and toplot[ii,jj,kk+1]:
                    pass
                else:
                    polys.append([[ii-0.5,jj+0.5,kk+0.5],[ii+0.5,jj+0.5,kk+0.5],
                                  [ii+0.5,jj-0.5,kk+0.5],[ii-0.5,jj-0.5,kk+0.5]]
                                 )
                    polycolors.append(colors[ii,jj,kk])
                if kk > 0 and toplot[ii,jj,kk-1]:
                    pass
                else:
                    polys.append([[ii-0.5,jj+0.5,kk-0.5],[ii+0.5,jj+0.5,kk-0.5],
                                  [ii+0.5,jj-0.5,kk-0.5],[ii-0.5,jj-0.5,kk-0.5]]
                                 )
                    polycolors.append(colors[ii,jj,kk])
                if jj < fs.shape[1]-1 and toplot[ii,jj+1,kk]:
                    pass
                else:
                    polys.append([[ii-0.5,jj+0.5,kk+0.5],[ii+0.5,jj+0.5,kk+0.5],
                                  [ii+0.5,jj+0.5,kk-0.5],[ii-0.5,jj+0.5,kk-0.5]]
                                 )
                    polycolors.append(colors[ii,jj,kk])
                if jj > 0 and toplot[ii,jj-1,kk]:
                    pass
                else:
                    polys.append([[ii-0.5,jj-0.5,kk+0.5],[ii+0.5,jj-0.5,kk+0.5],
                                  [ii+0.5,jj-0.5,kk-0.5],[ii-0.5,jj-0.5,kk-0.5]]
                                 )
                    polycolors.append(colors[ii,jj,kk])
                if ii < fs.shape[0]-1 and toplot[ii+1,jj,kk]:
                    pass
                else:
                    polys.append([[ii+0.5,jj-0.5,kk+0.5],[ii+0.5,jj+0.5,kk+0.5],
                                  [ii+0.5,jj+0.5,kk-0.5],[ii+0.5,jj-0.5,kk-0.5]]
                                 )
                    polycolors.append(colors[ii,jj,kk])
                if ii > 0 and toplot[ii-1,jj,kk]:
                    pass
                else:
                    polys.append([[ii-0.5,jj-0.5,kk+0.5],[ii-0.5,jj+0.5,kk+0.5],
                                  [ii-0.5,jj+0.5,kk-0.5],[ii-0.5,jj-0.5,kk-0.5]]
                                 )
                    polycolors.append(colors[ii,jj,kk])

    polycoll = mplot3d.art3d.Poly3DCollection(polys, facecolor=polycolors, 
                                              edgecolor='k', linewidths=0.5)
    ax.add_collection(polycoll)

    # Set the limits
    ax.set_xlim3d(-0.5,fs.shape[0]-0.5)
    ax.set_ylim3d(-0.5,fs.shape[1]-0.5)
    ax.set_zlim3d(-0.5,fs.shape[2]-0.5)

    ax.set_xlabel('aB', horizontalalignment='left')
    ax.set_ylabel('Ab', verticalalignment='bottom')
    ax.set_zlabel('AB', verticalalignment='bottom')

    ax.view_init(elev=30., azim=45)
    ax.colorbar()
    if show == True:
        plt.show()

def plot_3d_comp(model,data, resid_range=1, max_sum=None, max_index=None, show=True):
    """
    plots (model - data)/data
    resid_range - range of residuals (+/-)
    max_sum - slice spectrum on diagonal plane, plotting points closer to the origin than the plane
    max_index - only show entries this distance from one of the axes planes
    """
    model *= np.sum(data)/np.sum(model)
    resids = (model-data)/np.sqrt(data)
    for ii in range(len(resids)):
        for jj in range(len(resids)):
            for kk in range(len(resids)):
                if resids.mask[ii,jj,kk] == True:
                    continue
                elif np.isnan(resids[ii,jj,kk]) == True:
                    resids.mask[ii,jj,kk] = True
                elif np.isinf(resids[ii,jj,kk]) == True:
                    resids.mask[ii,jj,kk] = True
    
    fig = plt.figure()
    plt.clf()
    ax = mplot3d.Axes3D(fig)
    
    vmin = -resid_range
    vmax = resid_range
    
    resids = np.swapaxes(resids,0,2)
    
    toplot = np.logical_not(resids.mask)
    #toplot = np.logical_and(toplot, resids.data >= vmin)
    if max_sum != None:
        iis = np.where(toplot)[0]
        jjs = np.where(toplot)[1]
        kks = np.where(toplot)[2]
        for ll in range(len(iis)):
            ii = iis[ll]
            jj = jjs[ll]
            kk = kks[ll]
            if ii+jj+kk > max_sum:
                toplot[ii,jj,kk] = False
    if max_index != None:
        iis = np.where(toplot)[0]
        jjs = np.where(toplot)[1]
        kks = np.where(toplot)[2]
        for ll in range(len(iis)):
            ii = iis[ll]
            jj = jjs[ll]
            kk = kks[ll]
            if ii > max_index and jj > max_index and kk > max_index:
                toplot[ii,jj,kk] = False
    
    
    normalized = resids/(vmax-vmin) + .5
    #normalized = np.minimum(normalized, 1)
    # scrunch by a factor 
    # XXX: this is really hacky
    colors = plt.cm.RdBu(normalized)
    
    # We draw by calculating which faces are visible and including each as a
    # polygon.
    polys, polycolors = [],[]
    for ii in range(resids.shape[0]):
        for jj in range(resids.shape[1]):
            for kk in range(resids.shape[2]):
                if not toplot[ii,jj,kk]:
                    continue
                if kk < resids.shape[2]-1 and toplot[ii,jj,kk+1]:
                    pass
                else:
                    polys.append([[ii-0.5,jj+0.5,kk+0.5],[ii+0.5,jj+0.5,kk+0.5],
                                  [ii+0.5,jj-0.5,kk+0.5],[ii-0.5,jj-0.5,kk+0.5]]
                                 )
                    polycolors.append(colors[ii,jj,kk])
                if kk > 0 and toplot[ii,jj,kk-1]:
                    pass
                else:
                    polys.append([[ii-0.5,jj+0.5,kk-0.5],[ii+0.5,jj+0.5,kk-0.5],
                                  [ii+0.5,jj-0.5,kk-0.5],[ii-0.5,jj-0.5,kk-0.5]]
                                 )
                    polycolors.append(colors[ii,jj,kk])
                if jj < resids.shape[1]-1 and toplot[ii,jj+1,kk]:
                    pass
                else:
                    polys.append([[ii-0.5,jj+0.5,kk+0.5],[ii+0.5,jj+0.5,kk+0.5],
                                  [ii+0.5,jj+0.5,kk-0.5],[ii-0.5,jj+0.5,kk-0.5]]
                                 )
                    polycolors.append(colors[ii,jj,kk])
                if jj > 0 and toplot[ii,jj-1,kk]:
                    pass
                else:
                    polys.append([[ii-0.5,jj-0.5,kk+0.5],[ii+0.5,jj-0.5,kk+0.5],
                                  [ii+0.5,jj-0.5,kk-0.5],[ii-0.5,jj-0.5,kk-0.5]]
                                 )
                    polycolors.append(colors[ii,jj,kk])
                if ii < resids.shape[0]-1 and toplot[ii+1,jj,kk]:
                    pass
                else:
                    polys.append([[ii+0.5,jj-0.5,kk+0.5],[ii+0.5,jj+0.5,kk+0.5],
                                  [ii+0.5,jj+0.5,kk-0.5],[ii+0.5,jj-0.5,kk-0.5]]
                                 )
                    polycolors.append(colors[ii,jj,kk])
                if ii > 0 and toplot[ii-1,jj,kk]:
                    pass
                else:
                    polys.append([[ii-0.5,jj-0.5,kk+0.5],[ii-0.5,jj+0.5,kk+0.5],
                                  [ii-0.5,jj+0.5,kk-0.5],[ii-0.5,jj-0.5,kk-0.5]]
                                 )
                    polycolors.append(colors[ii,jj,kk])
    
    polycoll = mplot3d.art3d.Poly3DCollection(polys, facecolor=polycolors, 
                                              edgecolor='k', linewidths=0.5)
    ax.add_collection(polycoll)

    # Set the limits
    ax.set_xlim3d(-0.5,resids.shape[0]-0.5)
    ax.set_ylim3d(-0.5,resids.shape[1]-0.5)
    ax.set_zlim3d(-0.5,resids.shape[2]-0.5)

    ax.set_xlabel('aB', horizontalalignment='left')
    ax.set_ylabel('Ab', verticalalignment='bottom')
    ax.set_zlabel('AB', verticalalignment='bottom')
    
    ax.view_init(elev=30., azim=45)
    if show == True:
        plt.show()

