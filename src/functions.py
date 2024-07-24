import numpy as np
import numpy.linalg as LA

def getForces(km,dm,pos):
    """
    Calculate forces on atoms based on the spring constant matrix and displacement matrix.

    Parameters:
    - km (np.ndarray): Spring constant matrix.
    - dm (np.ndarray): Displacement matrix.
    - pos (list): List of atomic positions.

    Returns:
    - F (list): List of forces acting on each atom.
    """
	F = []
	for i in range(len(km)):
		Fi = np.array([0,0,0])
		for j in range(len(km[i])):
			if i != j:
				versor = pos[i] - pos[j]
				magnitude = LA.norm(versor)
				versor = versor/magnitude
				Fi = Fi - km[i][j]*versor*(magnitude-dm[i][j])
		F.append(Fi)
	return F

def Ebonds(Nbonds,Nangs,Noffplane,x, *args):
    """
    Energy calculation for bonds, angles, and off-plane interactions.

    Parameters:
    - Nbonds (int): Number of bonds.
    - Nangs (int): Number of angles.
    - Noffplane (int): Number of off-plane interactions.
    - x (np.ndarray): Array of bond distances and angles.
    - *args: Force constants for bonds and angles.

    Returns:
    - H (float): Total energy.
    """
    H = 0
    for i in range(Nbonds+Nangs+Noffplane):
        if i < Nbonds:
            H = H + 0.5*args[i]*(x[i])*(x[i])
        elif i >= Nbonds and i < Nbonds+Nangs:
            H = H + 0.5*args[i]*(x[i])*(x[i])
        else:
            H = H + 0.5*args[i]*(x[i])*(x[i])
    return H

def EbondsTwo(Nbonds,Nangs,x,b,a):
    """
    Energy calculation for classic bond an angle interactions using a simplified model.

    Parameters:
    - Nbonds (int): Number of bonds.
    - Nangs (int): Number of angles.
    - x (np.ndarray): Array of bond distances and angles.
    - b (float): Force constant for bonds.
    - a (float): Force constant for angles.

    Returns:
    - H (float): Total energy.
    """
    H = 0
    for i in range(Nbonds+Nangs):
        if i < Nbonds:
            H = H + 0.5*b*(x[i])*(x[i])
        else:
            H = H + 0.5*a*(x[i])*(x[i])
    return H

def EbondsThree(Nbonds,Nangs,Noffplane,x,b,a,o):
    """
    Energy calculation for classic bond, angle and offplane interactions using a simplified model.

    Parameters:
    - Nbonds (int): Number of bonds.
    - Nangs (int): Number of angles.
    - Noffplane (int): Number of off-plane interactions.
    - x (np.ndarray): Array of bond distances and angles.
    - b (float): Force constant for bonds.
    - a (float): Force constant for angles.
    - o (float): Force constant for off-plane interactions.

    Returns:
    - H (float): Total energy.
    """
    H = 0
    for i in range(Nbonds+Nangs+Noffplane):
        if i < Nbonds:
            H = H + 0.5*b*(x[i])*(x[i])
        elif i >= Nbonds and i < Nbonds+Nangs:
            H = H + 0.5*a*(x[i])*(x[i])
        else:
        	H = H + 0.5*o*(x[i])*(x[i])
    return H

def EbondsThree_proper(Nbonds,Nangs,Noffplane,Nproper,x,b,a,o):
    """
    Energy calculation for classic bond, angle and offplane interactions with proper dihedrals.

    Parameters:
    - Nbonds (int): Number of bonds.
    - Nangs (int): Number of angles.
    - Noffplane (int): Number of off-plane interactions.
    - Nproper (int): Number of proper dihedrals.
    - x (np.ndarray): Array of bond distances, angles, and dihedrals.
    - b (float): Force constant for bonds.
    - a (float): Force constant for angles.
    - o (float): Force constant for off-plane interactions.

    Returns:
    - H (float): Total energy.
    """
    H = 0
    for i in range(Nbonds+Nangs+Noffplane+Nproper):
        if i < Nbonds:
            H = H + 0.5*b*(x[i])*(x[i])
        elif i >= Nbonds and i < Nbonds+Nangs:
            H = H + 0.5*a*(x[i])*(x[i])
        elif i >= Nbonds+Nangs+Noffplane:
            H = H + 0.5*o*(x[i])*(x[i])
    return H

def min_delta(d,FMBD,fPW,r):
    """
    Minimizes the difference between MBD and pairwise (PW) forces using a decay rate correction.

    Parameters:
    - d (np.ndarray): Parameters for the force correction.
    - FMBD (np.ndarray): Forces calculated using MBD.
    - fPW (np.ndarray): Forces calculated using pairwise interactions.
    - r (np.ndarray): Distances between atoms.

    Returns:
    - ret (float): Normalized force difference.
    """
    FPW = np.array([0,0,0])
    for i in range(len(fPW)):
        if r[i] < 1:
            corr = 1
        else:
            corr=(r[i]/d[1])**(d[0]+d[2]*d[1]/r[i])
        FPW = FPW + fPW[i]*corr
    ret = LA.norm(FMBD-FPW)/LA.norm(FMBD)
    if ret > 100000 or np.isnan(ret) == True:
        ret = 100000
    return ret*100

def min_delta_vector(d,FMBD,fPW,r,density_versor):
    """
    Minimizes the difference between MBD and pairwise (PW) forces using a decay rate correction and direction correction.

    Parameters:
    - d (np.ndarray): Parameters for the force correction.
    - FMBD (np.ndarray): Forces calculated using MBD.
    - fPW (np.ndarray): Forces calculated using pairwise interactions.
    - r (np.ndarray): Distances between atoms.
    - density_versor (np.ndarray): Density versor for force correction.

    Returns:
    - ret (float): Normalized vector force difference.
    """
    FPW = np.array([0,0,0])
    for i in range(len(fPW)):
        corr=(r[i]/d[1])**(d[0]+d[2]*d[1]/r[i])
        if r[i] < 1:
            corr = 1
        FPW = FPW + fPW[i]*corr
    PW_versor = FPW/LA.norm(FPW)
    new_PW_versor = (1-np.abs(d[3]))*PW_versor+d[3]*density_versor
    new_PW_versor = new_PW_versor/LA.norm(new_PW_versor)
    FPW = LA.norm(FPW)*new_PW_versor
    ret = LA.norm(FMBD-FPW)/LA.norm(FMBD)
    if ret > 100000 or np.isnan(ret) == True:
        ret = 100000
    return ret*100

def min_delta_gauss_vector(d,FMBD,fPW,r,density_versor):
    """
    Minimizes the difference between MBD and pairwise (PW) forces using a decay rate correction and direction correction with gaussian damping.

    Parameters:
    - d (np.ndarray): Parameters for the force correction.
    - FMBD (np.ndarray): Forces calculated using MBD.
    - fPW (np.ndarray): Forces calculated using pairwise interactions.
    - r (np.ndarray): Distances between atoms.
    - density_versor (np.ndarray): Density versor for force correction.

    Returns:
    - ret (float): Normalized vector force difference with Gaussian correction.
    """
    FPW = np.array([0,0,0])
    for i in range(len(fPW)):
        corr=(1-np.exp(-(r[i]/d[2])**2))*(r[i]/d[1])**d[0]
        if r[i] < 1:
            corr = 1
        FPW = FPW + fPW[i]*corr
    PW_versor = FPW/LA.norm(FPW)
    new_PW_versor = (1-np.abs(d[3]))*PW_versor+d[3]*density_versor
    new_PW_versor = new_PW_versor/LA.norm(new_PW_versor)
    FPW = LA.norm(FPW)*new_PW_versor
    ret = LA.norm(FMBD-FPW)/LA.norm(FMBD)
    if ret > 100000 or np.isnan(ret) == True:
        ret = 100000
    return ret*100


def min_delta_force(d,FMBD,fPW,r):
    """
    Calculates the force difference between MBD and pairwise (PW) forces.

    Parameters:
    - d (np.ndarray): Parameters for the force correction.
    - FMBD (np.ndarray): Forces calculated using MBD.
    - fPW (np.ndarray): Forces calculated using pairwise interactions.
    - r (np.ndarray): Distances between atoms.

    Returns:
    - FPW (np.ndarray): Corrected pairwise forces.
    """
    FPW = np.array([0,0,0])
    for i in range(len(fPW)):
        corr=(r[i]/d[1])**(d[0]+d[2]*d[1]/r[i])
        if r[i] < 1:
            corr = 1
        FPW = FPW + fPW[i]*corr
    return FPW

def min_delta_salted(d,FMBD,fPW,r,e,w):
    """
    Minimizes the difference between MBD and pairwise (PW) forces with additional terms.

    Parameters:
    - d (np.ndarray): Parameters for the force correction.
    - FMBD (np.ndarray): Forces calculated using MBD.
    - fPW (np.ndarray): Forces calculated using pairwise interactions.
    - r (np.ndarray): Distances between atoms.
    - e (np.ndarray): Additional energy terms.
    - w (np.ndarray): Weights for the additional energy terms.

    Returns:
    - ret (float): Normalized force difference with additional terms.
    """
    return min_delta(d,FMBD,fPW,r) + w[0]*(d[0]-e[0])*(d[0]-e[0]) + w[1]*(d[1]-d[2])*(d[1]-d[2])

def min_delta_vector_salted(d,FMBD,fPW,r,density_versor,e,w):
    """
    Minimizes the vector difference between MBD and pairwise (PW) forces with additional terms.

    Parameters:
    - d (np.ndarray): Parameters for the force correction.
    - FMBD (np.ndarray): Forces calculated using MBD.
    - fPW (np.ndarray): Forces calculated using pairwise interactions.
    - r (np.ndarray): Distances between atoms.
    - density_versor (np.ndarray): Density versor for force correction.
    - e (np.ndarray): Additional energy terms.
    - w (np.ndarray): Weights for the additional energy terms.

    Returns:
    - ret (float): Normalized vector force difference with additional terms.
    """
    return min_delta_vector(d,FMBD,fPW,r,density_versor) + w[0]*(d[0]-e[0])*(d[0]-e[0]) + w[1]*(d[1]*d[2]/d[0])*(d[1]*d[2]/d[0]) + w[2]*(d[3])*(d[3])

def min_delta_gauss_vector_salted(d,FMBD,fPW,r,density_versor,e,w):
    """
    Minimizes the vector difference between MBD and pairwise (PW) forces using a Gaussian function with additional terms.

    Parameters:
    - d (np.ndarray): Parameters for the force correction.
    - FMBD (np.ndarray): Forces calculated using MBD.
    - fPW (np.ndarray): Forces calculated using pairwise interactions.
    - r (np.ndarray): Distances between atoms.
    - density_versor (np.ndarray): Density versor for force correction.
    - e (np.ndarray): Additional energy terms.
    - w (np.ndarray): Weights for the additional energy terms.

    Returns:
    - ret (float): Normalized vector force difference with Gaussian correction and additional terms.
    """
    return min_delta_gauss_vector(d,FMBD,fPW,r,density_versor) + w[0]*(d[0]-e[0])*(d[0]-e[0]) + w[1]*(d[2])*(d[2]) + w[2]*(d[3])*(d[3])
