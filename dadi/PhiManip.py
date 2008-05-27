import numpy

def phi_1D(xx, theta=1.0, gamma=0):
    """
    One-dimensional phi for a constant-sized population with selection.

    xx: one-dimensional grid of frequencies upon which phi is defined
    theta: scaled mutation rate, equal to 4*Nc * u, where u is the mutation 
           event rate per generation for the simulated locus.

    Returns a new phi array.
    """
    if gamma == 0:
        return phi_1D_snm(xx, theta)

    exp = numpy.exp
    phi = theta/(xx*(1-xx)) * (1-exp(-2*gamma*(1-xx)))/(1-exp(-2*gamma))
    if xx[0] == 0:
        phi[0] = phi[1]
    if xx[-1] == 1:
        limit = 2*gamma * exp(2*gamma)/(exp(2*gamma)-1)
        phi[-1] = limit
    return phi

def phi_1D_snm(xx, theta=1.0):
    """
    Standard neutral one-dimensional probability density.

    xx: one-dimensional grid of frequencies upon which phi is defined
    theta: scaled mutation rate, equal to 4*Nc * u, where u is the mutation 
           event rate per generation for the simulated locus.

    Returns a new phi array.
    """
    phi = theta/xx
    if xx[0] == 0:
        phi[0] = phi[1]
    return phi

def phi_1D_to_2D(xx, phi_1D):
    """
    Implement a one-to-two population split.

    xx: one-dimensional grid of frequencies upon which phi is defined
    phi1D: initial probability density

    Returns a new two-dimensional phi array.
    """
    pts = len(xx)
    phi_2D = numpy.zeros((pts, pts))
    for ii in range(1, pts-1):
        phi_2D[ii,ii] = phi_1D[ii] * 2/(xx[ii+1]-xx[ii-1])
    return phi_2D

def phi_2D_to_3D_split_2(xx, phi_2D):
    """
    Split population 2 into populations 2 and 3.

    xx: one-dimensional grid of frequencies upon which phi is defined
    phi2D: initial probability density

    Returns a new three-dimensional phi array.
    """
    pts = len(xx)
    phi_3D = numpy.zeros((pts, pts, pts))
    for jj in range(1,pts-1):
        phi_3D[:,jj,jj] = phi_2D[:,jj]*2/(xx[jj+1]-xx[jj-1])
    phi_3D[:,0,0] = phi_2D[:,0]*2/(xx[1] - xx[0])
    phi_3D[:,-1,-1] = phi_2D[:,-1]*2/(xx[-1] - xx[-2])
    return phi_3D

def phi_2D_to_3D_split_1(xx, phi_2D):
    """
    Split population 1 into populations 1 and 3.

    xx: one-dimensional grid of frequencies upon which phi is defined
    phi2D: initial probability density

    Returns a new three-dimensional phi array.
    """
    pts = len(xx)
    phi_3D = numpy.zeros((pts, pts, pts))
    for jj in range(1,pts-1):
        phi_3D[jj,:,jj] = phi_2D[jj,:]*2/(xx[jj+1]-xx[jj-1])
    phi_3D[0,:,0] = phi_2D[0,:]*2/(xx[1] - xx[0])
    phi_3D[-1,:,-1] = phi_2D[-1,:]*2/(xx[-1] - xx[-2])
    return phi_3D
