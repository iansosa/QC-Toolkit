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

def compressiontest_UHMWPE(vdw=None,struct_name=None,packing=[10,10,10],mbdpacking=[2,2,2]):
    if struct_name == None:
        if vdw == None:
            struct_name = "/UHMWPE/Primitive/UHMWPE_10-10-10.gen"
        elif vdw == "MBD":
            struct_name = "/UHMWPE/Primitive/UHMWPE_10-10-10_MBD333.gen"
        elif vdw == "TS":
            struct_name = "/UHMWPE/Primitive/UHMWPE_10-10-10_TS.gen"

    UHMWPE = structures.FromFile(struct_name)

    dcomp = -10.0/100.0
    for i in range(0,100):
        total_comp=i*dcomp
        ru = str(round(total_comp,2))
        if len(ru) == 4:
            ru = ru + '0'
        elif len(ru) == 3:
            ru = ru + '00'
        UHMWPE.SaveGeometry()
        UHMWPE.RunOptimize(vdw=vdw,static=None,read_charges=False,packing=packing,mbdpacking=mbdpacking,fixlengths="Y")
        UHMWPE.LoadGeometry("geo_end.gen")

        if vdw == None:
            UHMWPE.SaveGeometry("_"+ru,"UHMWPE")
        else:
            UHMWPE.SaveGeometry("_"+vdw+"_"+ru,"UHMWPE")
        UHMWPE.Displace_UC([0,dcomp,0])

# def compressiontest_UHMWPE_10x1x1(vdw=None,struct_name=None,packing=[2,5,5],mbdpacking=[2,3,3],du=0.1):

#     Nat = 5
#     R0 = 2.4

#     displ = []
#     for i in range(0,100):
#         u=i*du
#         ru = str(round(u,2))
#         if len(ru) == 4:
#             ru = ru + '0'
#         elif len(ru) == 3:
#             ru = ru + '00'
#         displ.append(ru)
#     print(displ)

#     if struct_name == None:
#         if vdw == None:
#             struct_name = "/UHMWPE/10x1x1/UHMWPE_2-5-5.gen"
#         elif vdw == "MBD":
#             struct_name = "/UHMWPE/10x1x1/UHMWPE_2-5-5_MBD233.gen"
#         elif vdw == "TS":
#             struct_name = "/UHMWPE/10x1x1/UHMWPE_2-5-5_TS.gen"


#     chain = structures.Sphere(Nat,R0)
#     chain.LoadGeometry(struct_name)
#     Nat = chain.Nat
#     chain.SaveGeometry()
#     chain.RunOptimize(vdw=vdw,static=None,read_charges=False,packing=packing,mbdpacking=mbdpacking,fixlengths="XYZ")
#     chain.LoadGeometry("geo_end.gen")
#     if vdw == None:
#         chain.SaveGeometry("_"+displ[0],"UHMWPE")
#     else:
#         chain.SaveGeometry("_"+vdw+"_"+displ[0],"UHMWPE")


#     for i in range(1,100):
#         chain.Displace_UC([du,0,0])
#         chain.SaveGeometry()
#         chain.RunOptimize(vdw=vdw,static=None,read_charges=False,packing=packing,mbdpacking=mbdpacking,fixlengths="XYZ")
#         chain.LoadGeometry("geo_end.gen")
#         u = i*du
#         if vdw == None:
#             chain.SaveGeometry("_"+displ[i],"UHMWPE")
#         else:
#             chain.SaveGeometry("_"+vdw+"_"+displ[i],"UHMWPE")


def UHMWPE_comp_stats(vdw=None, packing=[10,10,10],mbdpacking=[2,2,2]):
    dcomp = -5.0/100.0
    for i in range(0,100):
        total_comp = i*dcomp
        ru = str(round(total_comp,2))
        if len(ru) == 4:
            ru = ru + '0'
        elif len(ru) == 3:
            ru = ru + '00'

        if vdw == None:
            struct_name = "UHMWPE/Primitive/Compression/UHMWPE_comp_Y/_"+ru+".gen"
        elif vdw == "MBD":
            struct_name = "UHMWPE/Primitive/Compression/UHMWPE_comp_Y/_MBD_"+ru+".gen"
        elif vdw == "TS":
            struct_name = "UHMWPE/Primitive/Compression/UHMWPE_comp_Y/_TS_"+ru+".gen"

        UHMWPE = structures.FromFile(struct_name)
        UHMWPE.SaveGeometry()
        UHMWPE.RunStatic(vdw=vdw,packing=packing)
        E = UHMWPE.GetEnergy()

        with open('out/UHWPE_comp.txt', 'a') as f:
            for i in range(len(E)):
                f.write(ru+' '+str(E)+'\n')

compressiontest_UHMWPE()
UHMWPE_comp_stats()