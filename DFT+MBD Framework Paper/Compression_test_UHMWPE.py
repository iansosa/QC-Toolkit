import numpy as np
import numpy.linalg as LA
from mdhandler import Handler as MDH
from bondcalc import Bonds
import structures
import sys

def compressiontest_UHMWPE(vdw=None,struct_name=None,packing=[10,10,10],mbdpacking=[2,2,2]):

    Nat = 5
    R0 = 2.4

    u = 0
    du = 10.0/100.0

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

    if struct_name == None:
        if vdw == None:
            struct_name = "/UHMWPE/Primitive/UHMWPE_10-10-10.gen"
        elif vdw == "MBD":
            struct_name = "/UHMWPE/Primitive/UHMWPE_10-10-10_MBD333.gen"
        elif vdw == "PW":
            struct_name = "/UHMWPE/Primitive/UHMWPE_PW_2x2x2.gen"
        elif vdw == "TS":
            struct_name = "/UHMWPE/Primitive/UHMWPE_10-10-10_TS.gen"


    chain = structures.Sphere(Nat,R0)
    chain.LoadGeometry(struct_name)
    Nat = chain.Nat
    chain.SaveGeometry()
    chain.RunOptimize(vdw=vdw,static=None,read_charges=False,packing=packing,mbdpacking=mbdpacking,fixlengths="Y")
    chain.LoadGeometry("geo_end.gen")
    if vdw == None:
        chain.SaveGeometry("_"+displ[0],"UHMWPE")
    else:
        chain.SaveGeometry("_"+vdw+"_"+displ[0],"UHMWPE")


    for i in range(1,100):
        chain.Displace_UC([0,du,0])
        chain.SaveGeometry()
        chain.RunOptimize(vdw=vdw,static=None,read_charges=False,packing=packing,mbdpacking=mbdpacking,fixlengths="Y")
        chain.LoadGeometry("geo_end.gen")
        u = i*du
        if vdw == None:
            chain.SaveGeometry("_"+displ[i],"UHMWPE")
        else:
            chain.SaveGeometry("_"+vdw+"_"+displ[i],"UHMWPE")

def compressiontest_UHMWPE_10x1x1(vdw=None,struct_name=None,packing=[2,5,5],mbdpacking=[2,3,3],du=0.1):

    Nat = 5
    R0 = 2.4

    displ = []
    for i in range(0,100):
        u=i*du
        ru = str(round(u,2))
        if len(ru) == 4:
            ru = ru + '0'
        elif len(ru) == 3:
            ru = ru + '00'
        displ.append(ru)
    print(displ)

    if struct_name == None:
        if vdw == None:
            struct_name = "/UHMWPE/10x1x1/UHMWPE_2-5-5.gen"
        elif vdw == "MBD":
            struct_name = "/UHMWPE/10x1x1/UHMWPE_2-5-5_MBD233.gen"
        elif vdw == "TS":
            struct_name = "/UHMWPE/10x1x1/UHMWPE_2-5-5_TS.gen"


    chain = structures.Sphere(Nat,R0)
    chain.LoadGeometry(struct_name)
    Nat = chain.Nat
    chain.SaveGeometry()
    chain.RunOptimize(vdw=vdw,static=None,read_charges=False,packing=packing,mbdpacking=mbdpacking,fixlengths="XYZ")
    chain.LoadGeometry("geo_end.gen")
    if vdw == None:
        chain.SaveGeometry("_"+displ[0],"UHMWPE")
    else:
        chain.SaveGeometry("_"+vdw+"_"+displ[0],"UHMWPE")


    for i in range(1,100):
        chain.Displace_UC([du,0,0])
        chain.SaveGeometry()
        chain.RunOptimize(vdw=vdw,static=None,read_charges=False,packing=packing,mbdpacking=mbdpacking,fixlengths="XYZ")
        chain.LoadGeometry("geo_end.gen")
        u = i*du
        if vdw == None:
            chain.SaveGeometry("_"+displ[i],"UHMWPE")
        else:
            chain.SaveGeometry("_"+vdw+"_"+displ[i],"UHMWPE")

def compressiontest_UHMWPE_diag(vdw=None):

    Nat = 5
    R0 = 2.4

    u = 0
    du = 5.0/100.0

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
        struct_name = "/UHMWPE/UHMWPE_diag.gen"
    elif vdw == "MBD":
        struct_name = "/UHMWPE/UHMWPE_MBD_diag.gen"
    elif vdw == "PW":
        struct_name = "/UHMWPE/UHMWPE_PW_diag.gen"
    elif vdw == "TS":
        struct_name = "/UHMWPE/UHMWPE_TS_diag.gen"

    if vdw == None:
        ly = 4.775899536
        lz = 7.420825939
    if vdw == "MBD":
        ly = 4.413681356
        lz = 6.858008812
    if vdw == "PW":
        ly = 4.645894741
        lz = 7.218823584
    if vdw == "TS":
        ly = 4.456191901
        lz = 6.924061992

    c = 1/np.sqrt(ly*ly+lz*lz)

    chain = structures.Sphere(Nat,R0)
    chain.LoadGeometry(struct_name)
    Nat = chain.Nat
    chain.SaveGeometry()
    chain.RunOptimize(vdw=vdw,static=None,read_charges=False,fixlengths="Y")
    chain.LoadGeometry("geo_end.gen")
    if vdw == None:
        chain.SaveGeometry("_"+displ[0],"UHMWPE")
    else:
        chain.SaveGeometry("_"+vdw+"_"+displ[0],"UHMWPE")


    for i in range(1,100):
        chain.Displace_UC_2([0,c*ly*du,c*lz*du])
        chain.SaveGeometry()
        chain.RunOptimize(vdw=vdw,static=None,read_charges=False,fixlengths="Y")
        chain.LoadGeometry("geo_end.gen")
        u = i*du
        if vdw == None:
            chain.SaveGeometry("_"+displ[i],"UHMWPE")
        else:
            chain.SaveGeometry("_"+vdw+"_"+displ[i],"UHMWPE")


def UHMWPE_comp_stats(vdw=None, packing=[10,10,10],mbdpacking=[2,2,2]):

    Nat = 40
    R0 = 2.4
    chain = structures.Sphere(Nat,R0)

    u = 0
    du = -10.0/100.0

    displ = []
    # displ.append("0.00")
    for i in range(0,52):
        u=i*du
        ru = str(round(u,2))
        if len(ru) == 4:
            ru = ru + '0'
        elif len(ru) == 3:
            ru = ru + '00'
        displ.append(ru)
    print(displ)

    if vdw==None:
        E = []

        for i in range(len(displ)):
            
            chain.LoadGeometry("UHMWPE/Primitive/Compression/UHMWPE_comp_Y/_"+displ[i]+".gen")
            chain.SaveGeometry()
            chain.RunStatic(packing=packing)
            E.append(chain.GetEnergy())
            print(E)

        with open('out/UHWPE_E_diag.txt', 'w') as f:
            for i in range(len(E)):
                f.write(displ[i]+' '+str(E[i])+'\n')

    if vdw=="PW":
        E = []

        for i in range(len(displ)):
            
            chain.LoadGeometry("UHMWPE/Primitive/Compression/UHMWPE_comp_Y/_PW_"+displ[i]+".gen")
            chain.SaveGeometry()
            chain.RunStatic(vdw="PW",packing=packing)
            E.append(chain.GetEnergy())
            print(E)

        with open('out/UHWPE_E_diag-PW.txt', 'w') as f:
            for i in range(len(E)):
                f.write(displ[i]+' '+str(E[i])+'\n')


    if vdw=="TS":
        E = []

        for i in range(len(displ)):
            
            chain.LoadGeometry("UHMWPE/Primitive/Compression/UHMWPE_comp_Y/_TS_"+displ[i]+".gen")
            chain.SaveGeometry()
            chain.RunStatic(vdw="TS",packing=packing)
            E.append(chain.GetEnergy())
            print(E)

        with open('out/UHWPE_E_diag-TS.txt', 'w') as f:
            for i in range(len(E)):
                f.write(displ[i]+' '+str(E[i])+'\n')

    if vdw=="MBD":
        E = []

        for i in range(len(displ)):
            
            chain.LoadGeometry("UHMWPE/Primitive/Compression/UHMWPE_comp_Y/_MBD_"+displ[i]+".gen")
            chain.SaveGeometry()
            chain.RunStatic(vdw="MBD",packing=packing,mbdpacking=mbdpacking)
            E.append(chain.GetEnergy())
            print(E)

        with open('out/UHWPE_E_diag-MBD.txt', 'w') as f:
            for i in range(len(E)):
                f.write(displ[i]+' '+str(E[i])+'\n')

def UHMWPE_stress_stats(vdw=None, packing=[10,10,10], mbdpacking=[2,2,2]):

    Nat = 40
    R0 = 2.4
    chain = structures.Sphere(Nat,R0)

    u = 0
    du = -10.0/100.0

    displ = []
    for i in range(0,37):
        u=i*du
        ru = str(round(u,2))
        if len(ru) == 4:
            ru = ru + '0'
        elif len(ru) == 3:
            ru = ru + '00'
        displ.append(ru)
    print(displ)

    if vdw==None:
        for i in range(len(displ)):
            chain.LoadGeometry("UHMWPE/10x1x1/compression/novdw/_"+displ[i]+".gen")
            chain.SaveGeometry()
            chain.RunStatic(packing=packing)
            stress, eigenvectors = LA.eig(chain.GetStress())
            x = np.array([1,0,0])
            y = np.array([0,1,0])
            z = np.array([0,0,1])
            
            with open('out/UHMWPE_stress_novdw.txt', 'a') as f:
                f.write(displ[i]+' '+str(stress[0])+' '+str(stress[1])+' '+str(stress[2])+' '+str(np.abs(stress[0]*np.inner(x,eigenvectors[0])+stress[1]*np.inner(x,eigenvectors[1])+stress[2]*np.inner(x,eigenvectors[2])))+' '+str(np.abs(stress[0]*np.inner(y,eigenvectors[0])+stress[1]*np.inner(y,eigenvectors[1])+stress[2]*np.inner(y,eigenvectors[2])))+' '+str(np.abs(stress[0]*np.inner(z,eigenvectors[0])+stress[1]*np.inner(z,eigenvectors[1])+stress[2]*np.inner(z,eigenvectors[2])))+'\n')


    if vdw=="TS":
        for i in range(len(displ)):
            chain.LoadGeometry("UHMWPE/10x1x1/compression/TS/_TS_"+displ[i]+".gen")
            chain.SaveGeometry()
            chain.RunStatic(vdw="TS",packing=packing)
            stress, eigenvectors = LA.eig(chain.GetStress())
            x = np.array([1,0,0])
            y = np.array([0,1,0])
            z = np.array([0,0,1])

            with open('out/UHMWPE_stress_TS.txt', 'a') as f:
                f.write(displ[i]+' '+str(stress[0])+' '+str(stress[1])+' '+str(stress[2])+' '+str(np.abs(stress[0]*np.inner(x,eigenvectors[0])+stress[1]*np.inner(x,eigenvectors[1])+stress[2]*np.inner(x,eigenvectors[2])))+' '+str(np.abs(stress[0]*np.inner(y,eigenvectors[0])+stress[1]*np.inner(y,eigenvectors[1])+stress[2]*np.inner(y,eigenvectors[2])))+' '+str(np.abs(stress[0]*np.inner(z,eigenvectors[0])+stress[1]*np.inner(z,eigenvectors[1])+stress[2]*np.inner(z,eigenvectors[2])))+'\n')

    if vdw=="MBD":
        for i in range(len(displ)):
            chain.LoadGeometry("UHMWPE/10x1x1/compression/MBD/old/_MBD_"+displ[i]+".gen")
            chain.SaveGeometry()
            chain.RunStatic(vdw="MBD",packing=packing)
            stress, eigenvectors = LA.eig(chain.GetStress())
            x = np.array([1,0,0])
            y = np.array([0,1,0])
            z = np.array([0,0,1])

            with open('out/UHMWPE_stress_MBD.txt', 'a') as f:
                f.write(displ[i]+' '+str(stress[0])+' '+str(stress[1])+' '+str(stress[2])+' '+str(np.abs(stress[0]*np.inner(x,eigenvectors[0])+stress[1]*np.inner(x,eigenvectors[1])+stress[2]*np.inner(x,eigenvectors[2])))+' '+str(np.abs(stress[0]*np.inner(y,eigenvectors[0])+stress[1]*np.inner(y,eigenvectors[1])+stress[2]*np.inner(y,eigenvectors[2])))+' '+str(np.abs(stress[0]*np.inner(z,eigenvectors[0])+stress[1]*np.inner(z,eigenvectors[1])+stress[2]*np.inner(z,eigenvectors[2])))+'\n')


def pullout(vdw=None):
    Nat = 30
    R0 = 2.4

    u = 0
    du = 100.0/100.0

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
        struct_name = "/UHMWPE/Primitive/UHMWPE_5-5-5.gen"
    elif vdw == "MBD":
        struct_name = "/UHMWPE/Primitive/UHMWPE_10-10-10_MBD333.gen"
    elif vdw == "TS":
        struct_name = "/UHMWPE/Primitive/UHMWPE_5-5-5_TS.gen"
    chain = structures.Sphere(Nat,R0)
    chain.LoadGeometry(struct_name)
    chain.SaveGeometry()
    chain.RunOptimize(vdw=vdw,static=None,read_charges=False)
    chain.LoadGeometry("geo_end.gen")
    if vdw == None:
        chain.SaveGeometry("_"+displ[0],"pullout")
    else:
        chain.SaveGeometry("_"+vdw+"_"+displ[0],"pullout")

    for i in range(1,100):
        chain.Displace(9,[du,0,0])
        chain.SaveGeometry()
        chain.RunOptimize(vdw=vdw,static=None,read_charges=False)
        # chain.RunOptimize(vdw=vdw,static=[9,11],read_charges=False)
        chain.LoadGeometry("geo_end.gen")
        u = i*du
        if vdw == None:
            chain.SaveGeometry("_"+displ[i],"pullout")
        else:
            chain.SaveGeometry("_"+vdw+"_"+displ[i],"pullout")

def optimize555(vdw=None):
    Nat = 30
    R0 = 2.4
    if vdw == None:
        struct_name = "/UHMWPE/old/UHMWPE_2x2x2.gen"
    elif vdw == "MBD":
        struct_name = "/UHMWPE/old/UHMWPE_2x2x2.gen"
    elif vdw == "PW":
        struct_name = "/UHMWPE/old/UHMWPE_2x2x2.gen"
    elif vdw == "TS":
        struct_name = "/UHMWPE/old/UHMWPE_2x2x2.gen"


    chain = structures.Sphere(Nat,R0)
    chain.LoadGeometry(struct_name)
    chain.SaveGeometry()
    chain.RunOptimize(vdw=vdw,static=None,read_charges=False,packing=[5,5,5])
    chain.LoadGeometry("geo_end.gen")

    if vdw == None:
            savename = "UHMWPE_5-5-5_2x2x2"
    else:
            savename = vdw+"_UHMWPE_5-5-5_2x2x2"
    chain.SaveGeometry(savename,instruct=True,path="UHMWPE")
    imp.compressiontest_UHMWPE(vdw,"UHMWPE/"+savename+".gen",packing=[5,5,5])


def optimize101010(vdw=None):
    Nat = 30
    R0 = 2.4
    if vdw == None:
        struct_name = "/UHMWPE/UHMWPE.gen"
    elif vdw == "MBD":
        struct_name = "/UHMWPE/UHMWPE_MBD.gen"
    elif vdw == "PW":
        struct_name = "/UHMWPE/UHMWPE_PW.gen"
    elif vdw == "TS":
        struct_name = "/UHMWPE/UHMWPE_TS.gen"


    chain = structures.Sphere(Nat,R0)
    chain.LoadGeometry(struct_name)
    chain.SaveGeometry()
    chain.RunOptimize(vdw=vdw,static=None,read_charges=False,packing=[10,10,10])
    chain.LoadGeometry("geo_end.gen")

    if vdw == None:
            savename = "UHMWPE_10-10-10"
    else:
                savename = vdw+"_UHMWPE_10-10-10"
    chain.SaveGeometry(savename,instruct=True,path="UHMWPE")
    imp.compressiontest_UHMWPE(vdw,"UHMWPE/"+savename+".gen",packing=[10,10,10])

def optimize101010MBD(mbdpacking=[2,2,2]):

    Nat = 30
    R0 = 2.4
    struct_name = "/UHMWPE/UHMWPE_MBD.gen"
    vdw="MBD"

    chain = structures.Sphere(Nat,R0)
    chain.LoadGeometry(struct_name)
    chain.SaveGeometry()
    chain.RunOptimize(vdw=vdw,static=None,read_charges=False,packing=[10,10,10],mbdpacking=mbdpacking)
    chain.LoadGeometry("geo_end.gen")

    savename = vdw+"_UHMWPE_10-10-10"
    chain.SaveGeometry(savename,instruct=True,path="UHMWPE")
    imp.compressiontest_UHMWPE(vdw,"UHMWPE/"+savename+".gen",packing=[10,10,10],mbdpacking=mbdpacking)
