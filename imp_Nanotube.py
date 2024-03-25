import numpy as np
import numpy.linalg as LA
from mdhandler import Handler as MDH
from bondcalc import Bonds
import structures
import sys


def bucklingtest(vdw=None):

    Nat = 5
    R0 = 2.4

    u = 0
    du = 5.0/100.0

    displ = []
    # displ.append("0.00")
    for i in range(0,300):
        u=i*du
        ru = str(round(u,2))
        if len(ru) == 4:
            ru = ru + '0'
        elif len(ru) == 3:
            ru = ru + '00'
        displ.append(ru)
    print(displ)

    if vdw == None:
        struct_name = "/Nanotubes/Si.gen"
    elif vdw == "MBD":
        struct_name = "Nanotubes/buckling/Si/MBD/_MBD_-4.80.gen"
    elif vdw == "PW":
        struct_name = "Nanotubes/buckling/PW/geom_PW_-0.00.gen"
    elif vdw == "TS":
        struct_name = "Nanotubes/buckling/TS/_TS_-0.000.gen"

    chain = structures.Sphere(Nat,R0)
    chain.LoadGeometry(struct_name)
    Nat = chain.Nat

    SiNanotube = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,592,593,594,595,596,597,598,599,600,601,602,603,604,605,606,607,576,577,578,579,580,581,582,583,584,585,586,587,588,589,590,591]
    SiNanotube_leftwall = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,592,593,594,595,596,597,598,599,600,601,602,603,604,605,606,607,576,577,578,579,580,581,582,583,584,585,586,587,588,589,590,591]
    CNannotube = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,Nat-1,Nat-2,Nat-3,Nat-4,Nat-5,Nat-6,Nat-7,Nat-8,Nat-9,Nat-10,Nat-11,Nat-12,Nat-13,Nat-14,Nat-15,Nat-16]
    CNannotube_leftwall = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    chain.SaveGeometry()
    chain.RunOptimize(vdw=vdw,static=CNannotube,read_charges=False)
    chain.LoadGeometry()
    if vdw == None:
        chain.SaveGeometry("_"+displ[0],"buckling")
    else:
        chain.SaveGeometry("_"+vdw+"_"+displ[0],"buckling")


    

    for i in range(1,300):
        for j in range(len(CNannotube_leftwall)):
            chain.Displace(CNannotube_leftwall[j],[0,0,du])
            print(du)
        chain.SaveGeometry()
        chain.RunOptimize(vdw=vdw,static=CNannotube,read_charges=True)
        chain.LoadGeometry()
        if vdw == None:
            chain.SaveGeometry("_"+displ[i],"buckling")
        else:
            chain.SaveGeometry("_"+vdw+"_"+displ[i],"buckling")

def bendingtest(vdw=None):

    Nat = 5
    R0 = 2.4

    u = 0
    du = -5.0/100.0

    displ = []
    # displ.append("0.00")
    for i in range(0,100):
        u=i*du-4.00
        ru = str(round(u,2))
        if len(ru) == 4:
            ru = ru + '0'
        elif len(ru) == 3:
            ru = ru + '00'
        displ.append(ru)
    print(displ)

    if vdw == None:
        struct_name = "Nanotubes/Bending/BendingTest/_-4.00.gen"
    elif vdw == "MBD":
        struct_name = "Nanotubes/Bending/BendingTest/_MBD_-4.00.gen"
    elif vdw == "PW":
        struct_name = "Nanotubes/Bending/BendingTest/_PW_-4.00.gen"
    elif vdw == "TS":
        struct_name = "Nanotubes/Bending/BendingTest/_TS_-4.00.gen"

    chain = structures.Sphere(Nat,R0)
    chain.LoadGeometry(struct_name)
    Nat = chain.Nat
    chain.SaveGeometry()
    chain.RunOptimize(vdw=vdw,static=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],read_charges=False)
    chain.LoadGeometry()
    if vdw == None:
        chain.SaveGeometry("_"+displ[0],"buckling")
    else:
        chain.SaveGeometry("_"+vdw+"_"+displ[0],"buckling")

    for i in range(1,100):
        for j in range(16):
            chain.Displace(Nat-1-j,[du,0,0])
        chain.SaveGeometry()
        chain.RunOptimize(vdw=vdw,static=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],read_charges=True)
        chain.LoadGeometry()
        u = i*du
        if vdw == None:
            chain.SaveGeometry("_"+displ[i],"buckling")
        else:
            chain.SaveGeometry("_"+vdw+"_"+displ[i],"buckling")

def compressiontest(vdw=None):

    Nat = 5
    R0 = 2.4

    u = 0
    du = -5.0/100.0

    displ = []
    # displ.append("0.00")
    for i in range(0,100):
        u=i*du
        ru = str(round(u,2))
        if len(ru) == 4:
            ru = ru + '0'
        elif len(ru) == 3:
            ru = ru + '00'
        displ.append(ru)
    print(displ)

    if vdw == None:
        struct_name = "/Nanotubes/geom_5-10.gen"
    elif vdw == "MBD":
        struct_name = "/Nanotubes/geom_5-10_MBD.gen"
    elif vdw == "PW":
        struct_name = "/Nanotubes/geom_5-10_PW.gen"

    chain = structures.Sphere(Nat,R0)
    chain.LoadGeometry(struct_name)
    Nat = chain.Nat
    chain.SaveGeometry()
    chain.RunOptimize(vdw=vdw,static=[2,3,30,31,68,69,96,97,124,125,138,139,100,101,72,73,44,45,16,17],read_charges=False)
    chain.LoadGeometry()
    if vdw == None:
        chain.SaveGeometry("_"+displ[0],"nanotube")
    else:
        chain.SaveGeometry("_"+vdw+"_"+displ[0],"nanotube")


    for i in range(1,100):
        chain.Displace(2,[0,du,0])
        chain.Displace(3,[0,du,0])
        chain.Displace(30,[0,du,0])
        chain.Displace(31,[0,du,0])
        chain.Displace(68,[0,du,0])
        chain.Displace(69,[0,du,0])
        chain.Displace(96,[0,du,0])
        chain.Displace(97,[0,du,0])
        chain.Displace(124,[0,du,0])
        chain.Displace(125,[0,du,0])
        chain.Displace(138,[0,-du,0])
        chain.Displace(139,[0,-du,0])
        chain.Displace(100,[0,-du,0])
        chain.Displace(101,[0,-du,0])
        chain.Displace(72,[0,-du,0])
        chain.Displace(73,[0,-du,0])
        chain.Displace(44,[0,-du,0])
        chain.Displace(45,[0,-du,0])
        chain.Displace(16,[0,-du,0])
        chain.Displace(17,[0,-du,0])


        chain.SaveGeometry()
        chain.RunOptimize(vdw=vdw,static=[2,3,30,31,68,69,96,97,124,125,138,139,100,101,72,73,44,45,16,17],read_charges=True)
        chain.LoadGeometry()
        u = i*du
        if vdw == None:
            chain.SaveGeometry("_"+displ[i],"nanotube")
        else:
            chain.SaveGeometry("_"+vdw+"_"+displ[i],"nanotube")

def compressiontest_big(vdw=None):

    Nat = 5
    R0 = 2.4

    u = 0
    du = -5.0/100.0

    displ = []
    # displ.append("0.00")
    for i in range(0,100):
        u=i*du
        ru = str(round(u,2))
        if len(ru) == 4:
            ru = ru + '0'
        elif len(ru) == 3:
            ru = ru + '00'
        displ.append(ru)
    print(displ)

    if vdw == None:
        struct_name = "/Nanotubes/geom_15-30.gen"
    elif vdw == "MBD":
        struct_name = "/Nanotubes/geom_15-30_MBD.gen"
    elif vdw == "PW":
        struct_name = "/Nanotubes/geom_15-30_PW.gen"

    chain = structures.Sphere(Nat,R0)
    chain.LoadGeometry(struct_name)
    Nat = chain.Nat
    chain.SaveGeometry()
    chain.RunOptimize(vdw=vdw,static=[419,418,301,300,213,212,125,124,37,36,22,23,110,111,198,199,286,287,374,375],read_charges=False)
    chain.LoadGeometry()
    if vdw == None:
        chain.SaveGeometry("_"+displ[0],"nanotube")
    else:
        chain.SaveGeometry("_"+vdw+"_"+displ[0],"nanotube")


    for i in range(1,100):
        chain.Displace(419,[0,du,0])
        chain.Displace(418,[0,du,0])
        chain.Displace(301,[0,du,0])
        chain.Displace(300,[0,du,0])
        chain.Displace(213,[0,du,0])
        chain.Displace(212,[0,du,0])
        chain.Displace(125,[0,du,0])
        chain.Displace(124,[0,du,0])
        chain.Displace(37,[0,du,0])
        chain.Displace(36,[0,du,0])
        chain.Displace(22,[0,-du,0])
        chain.Displace(23,[0,-du,0])
        chain.Displace(110,[0,-du,0])
        chain.Displace(111,[0,-du,0])
        chain.Displace(198,[0,-du,0])
        chain.Displace(199,[0,-du,0])
        chain.Displace(286,[0,-du,0])
        chain.Displace(287,[0,-du,0])
        chain.Displace(374,[0,-du,0])
        chain.Displace(375,[0,-du,0])


        chain.SaveGeometry()
        chain.RunOptimize(vdw=vdw,static=[419,418,301,300,213,212,125,124,37,36,22,23,110,111,198,199,286,287,374,375],read_charges=True)
        chain.LoadGeometry()
        u = i*du
        if vdw == None:
            chain.SaveGeometry("_"+displ[i],"nanotube")
        else:
            chain.SaveGeometry("_"+vdw+"_"+displ[i],"nanotube")

def nanotube_graphene_vibration(vdw=None):

    if vdw == None:
        struct_name_graphene = "/Graphene/Graphene_C390_rotatedy.gen"
        struct_name_nanotube = "/Nanotubes/5-5-20_bent.gen"
        save_name = "NaGr_vib"
    elif vdw == "MBD":
        struct_name_graphene = "/Graphene/Graphene_C390-MBD_rotatedy.gen"
        struct_name_nanotube = "/Nanotubes/5-5-20-MBD_bent.gen"
        save_name = "NaGr_vib_MBD"
    elif vdw == "PW":
        struct_name_graphene = "/Graphene/Graphene_C390-PW_rotatedy.gen"
        struct_name_nanotube = "/Nanotubes/5-5-20-PW_bent.gen"
        save_name = "NaGr_vib_PW"
    elif vdw == "TS":
        struct_name_graphene = "/Graphene/Graphene_C390-TS_rotatedy.gen"
        struct_name_nanotube = "/Nanotubes/5-5-20-TS_bent.gen"
        save_name = "NaGr_vib_TS"


    Nat = 10
    R0 = 2.4

    geom = structures.Sphere(Nat,R0)
    geom.LoadGeometry(struct_name_graphene)

    geomall = structures.Sphere(Nat,R0)
    geomall.LoadGeometry(struct_name_nanotube)
    Nat_nanotube = geomall.Nat
    geomall.MoveAll([-90,8,58])
    geomall.add(geom)


    static_list_nanotube = [8, 9, 0, 1, 2, 3, 4, 5, 6, 7]
    static_list_graphene = [15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 40, 68, 96, 124, 152, 180, 208, 236, 264, 292, 320, 348, 376, 374, 389, 388, 387, 386, 385, 384, 383, 382, 381, 380, 379, 378, 377, 349, 321, 293, 265, 237, 209, 181, 153, 125, 97, 69, 41, 13]
    for i in range(len(static_list_graphene)):
        static_list_graphene[i]=static_list_graphene[i]+Nat_nanotube
    static_list = static_list_nanotube + static_list_graphene

    geomall.SaveGeometry()
    geomall.RunOptimize(vdw=vdw,static=static_list+[398, 399, 390, 391, 392, 393, 394, 395, 396, 397])
    geomall.LoadGeometry()
    geomall.SaveGeometry("opt_"+save_name)

    md = MDH(geomall,False)
    md.RunMD(steps=15000,temp=0,vdw=vdw,keepstationary=False,static=static_list)
    md.SaveEvolutionAs(save_name)

def nanotube_vibration(vdw=None):

    if vdw == None:
        struct_name_nanotube = "/Nanotubes/5-5-20_bent.gen"
        save_name = "Na_vib"
    elif vdw == "MBD":
        struct_name_nanotube = "/Nanotubes/5-5-20-MBD_bent.gen"
        save_name = "Na_vib_MBD"
    elif vdw == "PW":
        struct_name_nanotube = "/Nanotubes/5-5-20-PW_bent.gen"
        save_name = "Na_vib_PW"
    elif vdw == "TS":
        struct_name_nanotube = "/Nanotubes/5-5-20-TS_bent.gen"
        save_name = "Na_vib_TS"

    Nat = 10
    R0 = 2.4

    geomall = structures.Sphere(Nat,R0)
    geomall.LoadGeometry(struct_name_nanotube)


    static_list_nanotube = [8, 9, 0, 1, 2, 3, 4, 5, 6, 7]


    md = MDH(geomall,False)
    md.RunMD(steps=15000,temp=0,vdw=vdw,keepstationary=False,static=static_list_nanotube)
    md.SaveEvolutionAs(save_name)


def buckling_stats(vdw=None):

    Nat = 400
    R0 = 2.4


    du = 5.0/100.0
    u=0
    CNannotube_leftwall = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    # SiNanotube_leftwall = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,592,593,594,595,596,597,598,599,600,601,602,603,604,605,606,607,576,577,578,579,580,581,582,583,584,585,586,587,588,589,590,591]
    static_atoms = CNannotube_leftwall
    axis = 2 # 0 x, 1 y, 2 z

    if vdw == "PW":
        displ = []
        # displ.append('0.00')
        for i in range(0,77):
            u=i*du+0.00
            ru = str(round(u,2))
            if len(ru) == 2:
                ru = ru + '0'
            if len(ru) == 3:
                ru = ru + '0'
            # if len(ru) == 4:
            #     ru = ru + '0'
            displ.append(ru)
        print(displ)

        Fz_PW = np.zeros(len(displ))
        for i in range(len(displ)):
            name = "Nanotubes/buckling/Si/PW/_PW_-"+displ[i]+".gen"
            print(name)
            chain = structures.Sphere(Nat,R0)
            chain.LoadGeometry(name)
            chain.SaveGeometry()

            chain.RunStatic("PW",read_charges=False)
            F = chain.GetForces()
            for j in static_atoms:
                Fz_PW[i] = Fz_PW[i] + F[j][axis]
            print(Fz_PW)

        with open('out/Buckling_PW.txt', 'w') as f:
            for i in range(len(Fz_PW)):
                f.write(displ[i]+' '+str(Fz_PW[i])+'\n')

    if vdw == "TS":

        displ = []
        # displ.append("0.00")
        for i in range(0,139):
            u=i*du
            ru = str(round(u,2))
            if len(ru) == 4:
                ru = ru + '0'
            elif len(ru) == 3:
                ru = ru + '00'
            displ.append(ru)
        print(displ)

        Fz_TS = np.zeros(len(displ))
        for i in range(len(displ)):
            name = "Nanotubes/buckling/TS/_TS_"+displ[i]+".gen"
            print(name)
            chain = structures.Sphere(Nat,R0)
            chain.LoadGeometry(name)
            chain.SaveGeometry()

            chain.RunStatic("TS",read_charges=False)
            F = chain.GetForces()
            for j in static_atoms:
                Fz_TS[i] = Fz_TS[i] + F[j][axis]
            print(Fz_TS)

        with open('out/Buckling_TS.txt', 'w') as f:
            for i in range(len(Fz_TS)):
                f.write(displ[i]+' '+str(Fz_TS[i])+'\n')

    if vdw == "MBD":
        displ = []
        # displ.append('0.00')
        for i in range(64,65):
            u=i*du+0.00
            ru = str(round(u,2))
            if len(ru) == 2:
                ru = ru + '0'
            if len(ru) == 3:
                ru = ru + '0'
            # if len(ru) == 4:
            #     ru = ru + '0'
            displ.append(ru)
        print(displ)

        Fz_MBD = np.zeros(len(displ))
        for i in range(len(displ)):
            name = "Nanotubes/buckling/Si/MBD/_MBD_-"+displ[i]+".gen"
            print(name)
            chain = structures.Sphere(Nat,R0)
            chain.LoadGeometry(name)
            chain.SaveGeometry()

            chain.RunStatic("MBD",read_charges=False)
            F = chain.GetForces()
            for j in static_atoms:
                Fz_MBD[i] = Fz_MBD[i] + F[j][axis]
            print(Fz_MBD)

        with open('out/Buckling_MBD.txt', 'w') as f:
            for i in range(len(Fz_MBD)):
                f.write(displ[i]+' '+str(Fz_MBD[i])+'\n')


    if vdw == None:
        displ = []
        # displ.append('0.00')
        for i in range(0,77):
            u=i*du+0.00
            ru = str(round(u,2))
            if len(ru) == 2:
                ru = ru + '0'
            if len(ru) == 3:
                ru = ru + '0'
            # if len(ru) == 4:
            #     ru = ru + '0'
            displ.append(ru)
        print(displ)

        Fz = np.zeros(len(displ))
        for i in range(len(displ)):
            name = "Nanotubes/buckling/Si/novdw/_-"+displ[i]+".gen"
            print(name)
            chain = structures.Sphere(Nat,R0)
            chain.LoadGeometry(name)
            chain.SaveGeometry()

            chain.RunStatic(vdw=None,read_charges=False)
            F = chain.GetForces()
            for j in static_atoms:
                Fz[i] = Fz[i] + F[j][axis]
            print(Fz)

        with open('out/Buckling.txt', 'w') as f:
            for i in range(len(Fz)):
                f.write(displ[i]+' '+str(Fz[i])+'\n')


def nanotube_comp_stats():

    Nat = 40
    R0 = 2.4
    chain = structures.Sphere(Nat,R0)

    u = 0
    du = -5.0/100.0

    displ = []
    # displ.append("0.00")
    for i in range(0,20):
        u=i*du
        ru = str(round(u,2))
        if len(ru) == 4:
            ru = ru + '0'
        elif len(ru) == 3:
            ru = ru + '00'
        displ.append(ru)
    print(displ)

    indexes = [419,418,301,300,213,212,125,124,37,36]

    Fz_PW = np.zeros(len(displ))
    for i in range(len(displ)):
        chain.LoadGeometry("Nanotubes/Compression/compression_15-30/geom_PW_"+displ[i]+".gen")
        chain.SaveGeometry()
        chain.RunStatic("PW")
        F = chain.GetForces()
        for j in indexes:
            Fz_PW[i] = Fz_PW[i] + F[j][1]
        print(Fz_PW)

    with open('out/Nanotube_comp_PW.txt', 'w') as f:
        for i in range(len(Fz_PW)):
            f.write(displ[i]+' '+str(Fz_PW[i])+'\n')

    Fz_MBD = np.zeros(len(displ))
    for i in range(len(displ)):
        chain.LoadGeometry("Nanotubes/Compression/compression_15-30/geom_MBD_"+displ[i]+".gen")
        chain.SaveGeometry()
        chain.RunStatic("MBD")
        F = chain.GetForces()
        for j in indexes:
            Fz_MBD[i] = Fz_MBD[i] + F[j][1]
        print(Fz_MBD)

    with open('out/Nanotube_comp_MBD.txt', 'w') as f:
        for i in range(len(Fz_MBD)):
            f.write(displ[i]+' '+str(Fz_MBD[i])+'\n')

    Fz = np.zeros(len(displ))
    for i in range(len(displ)):
        chain.LoadGeometry("Nanotubes/Compression/compression_15-30/geom_"+displ[i]+".gen")
        chain.SaveGeometry()
        chain.RunStatic()
        F = chain.GetForces()
        for j in indexes:
            Fz[i] = Fz[i] + F[j][1]
        print(Fz)

    with open('out/Nanotube_comp.txt', 'w') as f:
        for i in range(len(Fz)):
            f.write(displ[i]+' '+str(Fz[i])+'\n')
