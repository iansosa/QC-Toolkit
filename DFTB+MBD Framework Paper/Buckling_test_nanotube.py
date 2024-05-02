import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import numpy.linalg as LA
from mdhandler import Handler as MDH
from bondcalc import Bonds
import structures
import sys

def buckling_simulation(vdw=None):
    if vdw == None:
        struct_name = "Nanotubes/buckling/novdw/original.gen"
    elif vdw == "MBD":
        struct_name = "Nanotubes/buckling/MBD/original.gen"
    elif vdw == "TS":
        struct_name = "Nanotubes/buckling/TS/original.gen"

    Nanotube = structures.FromFile(struct_name)
    Nat = Nanotube.Nat
    print(Nat)
    Nanotube_boundaries = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,Nat-1,Nat-2,Nat-3,Nat-4,Nat-5,Nat-6,Nat-7,Nat-8,Nat-9,Nat-10,Nat-11,Nat-12,Nat-13,Nat-14,Nat-15,Nat-16]
    Nanotube_left_boundary = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    Nanotube.SaveGeometry()

    dcomp = 5.0/100.0
    for i in range(0,100):
        total_comp = i*dcomp
        ru = str(round(total_comp,2))
        if len(ru) == 3:
            ru = ru + '0'

        Nanotube.SaveGeometry()
        Nanotube.RunOptimize(vdw=vdw,static=Nanotube_boundaries,read_charges=False)
        Nanotube.LoadGeometry()

        if vdw == None:
            Nanotube.SaveGeometry("geom_"+ru,"buckling")
        else:
            Nanotube.SaveGeometry("geom_"+vdw+"_"+ru,"buckling")
        
        for j in range(len(Nanotube_left_boundary)):
            Nanotube.Displace(Nanotube_left_boundary[j],[0,0,dcomp])

def buckling_statistics(vdw=None):
    Nanotube_left_boundary = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    axis = 2 # 0 x, 1 y, 2 z

    dcomp = 5.0/100.0
    for i in range(0,100):
        total_comp = i*dcomp
        ru = str(round(total_comp,2))
        if len(ru) == 3:
            ru = ru + '0'

        if vdw == None:
            struct_name = "Nanotubes/buckling/novdw/geom_"+ru+".gen"
        elif vdw == "MBD":
            struct_name = "Nanotubes/buckling/MBD/geom_MBD_"+ru+".gen"
        elif vdw == "TS":
            struct_name = "Nanotubes/buckling/TS/geom_TS_"+ru+".gen"

        
        Nanotube = structures.FromFile(struct_name)
        Nanotube.SaveGeometry()
        Nanotube.RunStatic(vdw=vdw,read_charges=False)
        F = Nanotube.GetForces()
        F_total = 0
        for j in Nanotube_left_boundary:
            F_total = F_total + F[j][axis]
        with open('out/Buckling.txt', 'a') as f:
            f.write(ru+' '+str(F_total)+'\n')

buckling_simulation()
buckling_statistics()