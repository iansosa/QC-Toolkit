import numpy as np
import numpy.linalg as LA

def getForces(km,dm,pos):
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
    H = 0
    for i in range(Nbonds+Nangs):
        if i < Nbonds:
            H = H + 0.5*b*(x[i])*(x[i])
        else:
            H = H + 0.5*a*(x[i])*(x[i])
    return H

def EbondsThree(Nbonds,Nangs,Noffplane,x,b,a,o):
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
    FPW = np.array([0,0,0])
    for i in range(len(fPW)):
        corr=(r[i]/d[1])**(d[0]+d[2]*d[1]/r[i])
        if r[i] < 1:
            corr = 1
        FPW = FPW + fPW[i]*corr
    ret = LA.norm(FMBD-FPW)/LA.norm(FMBD)
    if ret > 100000 or np.isnan(ret) == True:
        ret = 100000
    return ret*100

def min_delta_vector(d,FMBD,fPW,r,density_versor):
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
    FPW = np.array([0,0,0])
    for i in range(len(fPW)):
        corr=(r[i]/d[1])**(d[0]+d[2]*d[1]/r[i])
        if r[i] < 1:
            corr = 1
        FPW = FPW + fPW[i]*corr
    return FPW

def min_delta_salted(d,FMBD,fPW,r,e,w):
    return min_delta(d,FMBD,fPW,r) + w[0]*(d[0]-e[0])*(d[0]-e[0]) + w[1]*(d[1]-d[2])*(d[1]-d[2])

def min_delta_vector_salted(d,FMBD,fPW,r,density_versor,e,w):
    return min_delta_vector(d,FMBD,fPW,r,density_versor) + w[0]*(d[0]-e[0])*(d[0]-e[0]) + w[1]*(d[1]*d[2]/d[0])*(d[1]*d[2]/d[0]) + w[2]*(d[3])*(d[3])

def min_delta_gauss_vector_salted(d,FMBD,fPW,r,density_versor,e,w):
    return min_delta_gauss_vector(d,FMBD,fPW,r,density_versor) + w[0]*(d[0]-e[0])*(d[0]-e[0]) + w[1]*(d[2])*(d[2]) + w[2]*(d[3])*(d[3])
