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

def min_delta_force(d,FMBD,fPW,r):
    FPW = np.array([0,0,0])
    for i in range(len(fPW)):
        corr=(r[i]/d[1])**(d[0]+d[2]*d[1]/r[i])
        if r[i] < 1:
            corr = 1
        FPW = FPW + fPW[i]*corr
    return FPW

def min_delta_salted(d,FMBD,fPW,r):
    return min_delta(d,FMBD,fPW,r) + 1*(d[0]-1.5)*(d[0]-1.5) + 0.2*(d[0]+d[2])*(d[0]+d[2])

