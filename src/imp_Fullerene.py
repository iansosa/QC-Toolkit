import numpy as np
import numpy.linalg as LA
from mdhandler import Handler as MDH
from bondcalc import Bonds
import structures
import sys

def decompose(T,name,vdw=None):

    Nat = 30
    R0 = 2.4

    if vdw == None:
        struct_name = "fullerenes/"+name+".gen"
    else:
        struct_name = "fullerenes/"+name+"_"+vdw+".gen"
            
    geom = structures.Sphere(Nat,R0)
    geom.LoadGeometry(struct_name)
    md = MDH(geom,False)
    if vdw == None:
        path = "../TRAVIS/exe/trajectories/Fullerene"+name+"_"+str(T)+".xyz"
    else:
        path = "../TRAVIS/exe/trajectories/Fullerene"+name+"_"+vdw+"_"+str(T)+".xyz"
    md.LoadEvolution(path)

    if vdw == None:
        save_path = "Decomposed/"+name+"_"+str(T)
        save_name = "Fullerene"+name+"_"+str(T)
    else:
        save_path = "Decomposed/"+name+"_"+str(T)+"_"+vdw
        save_name = "Fullerene"+name+"_"+vdw+"_"+str(T)
        
    md.DecomposeTrajectory(save_name,save_path)

def heat_capacity(T,vdw=None,min_steps=1000):
    Nat = 30
    R0 = 2.4
    if vdw == None:
        struct_name = "fullerenes/C720.gen"
    else:
        struct_name = "fullerenes/C720_"+vdw+".gen"
        
    geom = structures.Sphere(Nat,R0)
    geom.LoadGeometry("fullerenes/C720.gen")
    md = MDH(geom,False)
    if vdw == None:
        path = "../TRAVIS/exe/trajectories/FullereneC720_"+str(T)+".xyz"
    else:
        path = "../TRAVIS/exe/trajectories/FullereneC720_"+vdw+"_"+str(T)+".xyz"
    md.LoadEvolution(path)

    kb = 0.000003167 #boltzmann constant in H/K

    ken = md.ComputeKenergy()
    ven = md.ComputeVenergy(vdw=vdw)
    Ten = []
    for i in range(len(ken)):
        Ten.append(ken[i]+ven[i])


    Aven = np.zeros(len(Ten))
    for i in range(len(Ten)):
        if min_steps < i+1:
            for j in range(min_steps,i+1):
                Aven[i] = Aven[i] + Ten[j]/(i+1-min_steps)
    Avendiff = np.zeros(len(Ten))

    for i in range(len(Ten)):
        if min_steps < i+1:
            for j in range(min_steps,i+1):
                Avendiff[i] = Avendiff[i] + (Ten[j] - Aven[i])*(Ten[j] - Aven[i])/(i+1-min_steps)
    Cv = []

    for i in range(len(Ten)):
        Cv.append(Avendiff[i]/(kb*T*T)) #heat capacity in H/K

    if vdw == None:
        save_name = "out/Cv_F720_"+str(T)+".txt"
    else:
        save_name = "out/Cv_F720_"+vdw+"_"+str(T)+".txt"

    with open(save_name, 'w') as f:
        for i in range(len(Cv)):
            f.write(str(i)+' '+str(ken[i])+' '+str(ven[i])+' '+str(Ten[i])+' '+str(Aven[i])+' '+str(Avendiff[i])+' '+str(Cv[i])+'\n')

def long_md(name,T,vdw=None):
    struct_name = name+".gen"

    geom = structures.FromFile(struct_name)
    geom.SaveGeometry()
    md = MDH(geom,False)
    md.RunMD(steps=1500000,temp=T,vdw=vdw,keepstationary=True)