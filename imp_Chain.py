import numpy as np
import numpy.linalg as LA
from mdhandler import Handler as MDH
from bondcalc import Bonds
import structures
import sys
import scipy.stats as ss

def StaticOverEvolve_Chains(temp,Length,distance):
    if distance <0:
        print("distance should be positive")
        sys.exit()
    Nat=Length
    R0=2.4
    Chain1 = structures.Chain(Nat,R0)
    Chain1.SaveGeometry()
    Chain1.RunOptimize()
    Chain1.LoadGeometry()

    Chain2 = structures.Chain(Nat,R0)
    Chain2.SaveGeometry()
    Chain2.RunOptimize()
    Chain2.LoadGeometry()

    Chain2.MoveAll([0,0,distance])

    Chain1.add(Chain2)

    md = MDH(Chain1,False)
    md.RunMD(5000,temp,[0,Length-1,Length,2*Length-1])
    ForcesMBD = md.GetForcesSOE("MBD")
    ForcesPW = md.GetForcesSOE("PW")
    ForcesShort = md.GetForcesSOE()

    with open('out/SOE_'+str(temp)+'.txt', 'w') as f:
        for i in range(len(ForcesShort)):
            F_lower_MBD = 0
            F_lower_PW = 0
            F_lower_Short = 0
            for k in range(Length):
                F_lower_MBD = F_lower_MBD + ForcesMBD[i][k][2]
                F_lower_PW = F_lower_PW + ForcesPW[i][k][2]
                F_lower_Short = F_lower_Short + ForcesShort[i][k][2]
            F_upper_MBD = 0
            F_upper_PW = 0
            F_upper_Short = 0
            for k in range(Length,2*Length):
                F_upper_MBD = F_upper_MBD + ForcesMBD[i][k][2]
                F_upper_PW = F_upper_PW + ForcesPW[i][k][2]
                F_upper_Short = F_upper_Short + ForcesShort[i][k][2]
            f.write(str(i)+ " " +str(F_upper_MBD-F_upper_Short)+ " "+ str(F_lower_MBD-F_lower_Short)+ " " +str(F_upper_PW-F_upper_Short)+ " "+ str(F_lower_PW-F_lower_Short)+ " " +str(F_upper_Short)+ " "+ str(F_lower_Short)+"\n")

def CorrelationOverEvolve_Chains(temp,Length,distance,vdw):
    if distance <0:
        print("distance should be positive")
        sys.exit()
    Nat=Length
    R0=2.4
    Chain1 = structures.ChainCapped(Nat,R0)
    Chain1.SaveGeometry()
    # Chain1.RunOptimize()
    Chain1.LoadGeometry()

    Chain2 = structures.ChainCapped(Nat,R0)
    Chain2.SaveGeometry()
    # Chain2.RunOptimize()
    Chain2.LoadGeometry()

    Chain2.MoveAll([0,0,distance])
    Chain1.add(Chain2)

    Navg=100
    corrxx = np.zeros(Length)
    corryy = np.zeros(Length)
    corrzz = np.zeros(Length)

    corrxy = np.zeros(Length)
    corrxz = np.zeros(Length)
    corryz = np.zeros(Length)

    corrxx_abs = np.zeros(Length)
    corryy_abs = np.zeros(Length)
    corrzz_abs = np.zeros(Length)

    corrxy_abs = np.zeros(Length)
    corrxz_abs = np.zeros(Length)
    corryz_abs = np.zeros(Length)

    for k in range(Navg):
        print(str(k)+"/"+str(Navg))
        md = MDH(Chain1,False)
        md.RunMD(steps=5000,temp=temp,static=[0,Length-1,Length,2*Length-1],vdw=vdw)
        md.LoadEvolution()
        evolution = md.evolution
        for j in range(Length):
            v_lower_x = []
            v_lower_y = []
            v_lower_z = []
            v_lower_x_abs = []
            v_lower_y_abs = []
            v_lower_z_abs = []
            for i in range(len(evolution)):
                v_lower_x.append(evolution[i][j][0])
                v_lower_y.append(evolution[i][j][1])
                v_lower_z.append(evolution[i][j][2])
                v_lower_x_abs.append(np.abs(evolution[i][j][0]))
                v_lower_y_abs.append(np.abs(evolution[i][j][1]))
                v_lower_z_abs.append(np.abs(evolution[i][j][2]))
            v_lower_x=np.array(v_lower_x)
            v_lower_y=np.array(v_lower_y)
            v_lower_z=np.array(v_lower_z)
            v_lower_x_abs=np.array(v_lower_x_abs)
            v_lower_y_abs=np.array(v_lower_y_abs)
            v_lower_z_abs=np.array(v_lower_z_abs)

            v_upper_x = []
            v_upper_y = []
            v_upper_z = []
            v_upper_x_abs = []
            v_upper_y_abs = []
            v_upper_z_abs = []
            for i in range(len(evolution)):
                v_upper_x.append(evolution[i][j+Length][0])
                v_upper_y.append(evolution[i][j+Length][1])
                v_upper_z.append(evolution[i][j+Length][2])
                v_upper_x_abs.append(np.abs(evolution[i][j+Length][0]))
                v_upper_y_abs.append(np.abs(evolution[i][j+Length][1]))
                v_upper_z_abs.append(np.abs(evolution[i][j+Length][2]))
            v_upper_x=np.array(v_upper_x)
            v_upper_y=np.array(v_upper_y)
            v_upper_z=np.array(v_upper_z)
            v_upper_x_abs=np.array(v_upper_x_abs)
            v_upper_y_abs=np.array(v_upper_y_abs)
            v_upper_z_abs=np.array(v_upper_z_abs)

            v_lower = []
            v_lower.append(v_lower_x)
            v_lower.append(v_lower_y)
            v_lower.append(v_lower_z)
            v_lower.append(v_lower_x_abs)
            v_lower.append(v_lower_y_abs)
            v_lower.append(v_lower_z_abs)
            v_upper = []
            v_upper.append(v_upper_x)
            v_upper.append(v_upper_y)
            v_upper.append(v_upper_z)
            v_upper.append(v_upper_x_abs)
            v_upper.append(v_upper_y_abs)
            v_upper.append(v_upper_z_abs)

            corrxx[j]=corrxx[j]+np.corrcoef(v_lower,v_upper)[0,6]/Navg
            corryy[j]=corryy[j]+np.corrcoef(v_lower,v_upper)[1,7]/Navg
            corrzz[j]=corrzz[j]+np.corrcoef(v_lower,v_upper)[2,8]/Navg

            corrxy[j]=corrxy[j]+np.corrcoef(v_lower,v_upper)[0,7]/Navg
            corrxz[j]=corrxz[j]+np.corrcoef(v_lower,v_upper)[0,8]/Navg
            corryz[j]=corryz[j]+np.corrcoef(v_lower,v_upper)[1,8]/Navg

            corrxx_abs[j]=corrxx_abs[j]+np.corrcoef(v_lower,v_upper)[3,9]/Navg
            corryy_abs[j]=corryy_abs[j]+np.corrcoef(v_lower,v_upper)[4,10]/Navg
            corrzz_abs[j]=corrzz_abs[j]+np.corrcoef(v_lower,v_upper)[5,11]/Navg

            corrxy_abs[j]=corrxy_abs[j]+np.corrcoef(v_lower,v_upper)[3,10]/Navg
            corrxz_abs[j]=corrxz_abs[j]+np.corrcoef(v_lower,v_upper)[3,11]/Navg
            corryz_abs[j]=corryz_abs[j]+np.corrcoef(v_lower,v_upper)[4,11]/Navg

    with open('out/corr_'+str(temp)+'.txt', 'w') as f:
        for i in range(len(corrxx)):
            f.write(str(i)+ " " +str(corrxx[i])+" " +str(corryy[i])+" " +str(corrzz[i])+" " +str(corrxy[i])+" " +str(corrxz[i])+" " +str(corryz[i])+" " +str(corrxx_abs[i])+" " +str(corryy_abs[i])+" " +str(corrzz_abs[i])+" " +str(corrxy_abs[i])+" " +str(corrxz_abs[i])+" " +str(corryz_abs[i])+"\n")

def CorrelationOverEvolve_Chain(temp,Length):
    Nat=Length
    R0=2.4
    Chain1 = structures.Chain(Nat,R0)
    Chain1.SaveGeometry()
    Chain1.RunOptimize()
    Chain1.LoadGeometry()

    Navg=100
    corrxx = np.zeros(Length)
    corryy = np.zeros(Length)
    corrzz = np.zeros(Length)

    corrxy = np.zeros(Length)
    corrxz = np.zeros(Length)
    corryz = np.zeros(Length)

    corrxx_abs = np.zeros(Length)
    corryy_abs = np.zeros(Length)
    corrzz_abs = np.zeros(Length)

    corrxy_abs = np.zeros(Length)
    corrxz_abs = np.zeros(Length)
    corryz_abs = np.zeros(Length)

    for k in range(Navg):
        print(str(k)+"/"+str(Navg))
        md = MDH(Chain1,False)
        md.RunMD(5000,temp,[0,Length-1])
        md.LoadEvolution()
        evolution = md.evolution
        for j in range(Length-1):
            v_lower_x = []
            v_lower_y = []
            v_lower_z = []
            v_lower_x_abs = []
            v_lower_y_abs = []
            v_lower_z_abs = []
            for i in range(len(evolution)):
                v_lower_x.append(evolution[i][j][0])
                v_lower_y.append(evolution[i][j][1])
                v_lower_z.append(evolution[i][j][2])
                v_lower_x_abs.append(np.abs(evolution[i][j][0]))
                v_lower_y_abs.append(np.abs(evolution[i][j][1]))
                v_lower_z_abs.append(np.abs(evolution[i][j][2]))
            v_lower_x=np.array(v_lower_x)
            v_lower_y=np.array(v_lower_y)
            v_lower_z=np.array(v_lower_z)
            v_lower_x_abs=np.array(v_lower_x_abs)
            v_lower_y_abs=np.array(v_lower_y_abs)
            v_lower_z_abs=np.array(v_lower_z_abs)

            v_upper_x = []
            v_upper_y = []
            v_upper_z = []
            v_upper_x_abs = []
            v_upper_y_abs = []
            v_upper_z_abs = []
            for i in range(len(evolution)):
                v_upper_x.append(evolution[i][j+1][0])
                v_upper_y.append(evolution[i][j+1][1])
                v_upper_z.append(evolution[i][j+1][2])
                v_upper_x_abs.append(np.abs(evolution[i][j+1][0]))
                v_upper_y_abs.append(np.abs(evolution[i][j+1][1]))
                v_upper_z_abs.append(np.abs(evolution[i][j+1][2]))
            v_upper_x=np.array(v_upper_x)
            v_upper_y=np.array(v_upper_y)
            v_upper_z=np.array(v_upper_z)
            v_upper_x_abs=np.array(v_upper_x_abs)
            v_upper_y_abs=np.array(v_upper_y_abs)
            v_upper_z_abs=np.array(v_upper_z_abs)

            v_lower = []
            v_lower.append(v_lower_x)
            v_lower.append(v_lower_y)
            v_lower.append(v_lower_z)
            v_lower.append(v_lower_x_abs)
            v_lower.append(v_lower_y_abs)
            v_lower.append(v_lower_z_abs)
            v_upper = []
            v_upper.append(v_upper_x)
            v_upper.append(v_upper_y)
            v_upper.append(v_upper_z)
            v_upper.append(v_upper_x_abs)
            v_upper.append(v_upper_y_abs)
            v_upper.append(v_upper_z_abs)

            corrxx[j]=corrxx[j]+np.corrcoef(v_lower,v_upper)[0,6]/Navg
            corryy[j]=corryy[j]+np.corrcoef(v_lower,v_upper)[1,7]/Navg
            corrzz[j]=corrzz[j]+np.corrcoef(v_lower,v_upper)[2,8]/Navg

            corrxy[j]=corrxy[j]+np.corrcoef(v_lower,v_upper)[0,7]/Navg
            corrxz[j]=corrxz[j]+np.corrcoef(v_lower,v_upper)[0,8]/Navg
            corryz[j]=corryz[j]+np.corrcoef(v_lower,v_upper)[1,8]/Navg

            corrxx_abs[j]=corrxx_abs[j]+np.corrcoef(v_lower,v_upper)[3,9]/Navg
            corryy_abs[j]=corryy_abs[j]+np.corrcoef(v_lower,v_upper)[4,10]/Navg
            corrzz_abs[j]=corrzz_abs[j]+np.corrcoef(v_lower,v_upper)[5,11]/Navg

            corrxy_abs[j]=corrxy_abs[j]+np.corrcoef(v_lower,v_upper)[3,10]/Navg
            corrxz_abs[j]=corrxz_abs[j]+np.corrcoef(v_lower,v_upper)[3,11]/Navg
            corryz_abs[j]=corryz_abs[j]+np.corrcoef(v_lower,v_upper)[4,11]/Navg

    with open('out/corr_'+str(temp)+'.txt', 'w') as f:
        for i in range(len(corrxx)):
            f.write(str(i)+ " " +str(corrxx[i])+" " +str(corryy[i])+" " +str(corrzz[i])+" " +str(corrxy[i])+" " +str(corrxz[i])+" " +str(corryz[i])+" " +str(corrxx_abs[i])+" " +str(corryy_abs[i])+" " +str(corrzz_abs[i])+" " +str(corrxy_abs[i])+" " +str(corrxz_abs[i])+" " +str(corryz_abs[i])+"\n")


def hessian_chain_rings(Nat,struct):
    R0 = 2.4
    if struct == "chain":
        chain = structures.Chain(Nat,R0)
        chain.SaveGeometry()
    else:
        chain = structures.Ring(Nat,R0)
        chain.SaveGeometry()

    bonder = Bonds(chain,False,False)
    bonder.CalcSaveHessianCompOur(struct+"_"+str(Nat))


def ForceDistr(vdw):
    Length = 30
    distance = 10

    Nat=Length
    R0=2.4
    Chain1 = structures.ChainCapped(Nat,R0)
    Chain1.SaveGeometry()
    # Chain1.RunOptimize()
    Chain1.LoadGeometry()

    Chain2 = structures.ChainCapped(Nat,R0)
    Chain2.SaveGeometry()
    # Chain2.RunOptimize()
    Chain2.LoadGeometry()

    Chain2.MoveAll([0,0,distance])
    Chain1.add(Chain2)

    # Chain1.RunOptimize(vdw=vdw,static=[0,Length-1,Length,2*Length-1])
    Chain1.LoadGeometry()
    Chain1.SaveGeometry()

    md = MDH(Chain1,False)
    if vdw == None:
        md.LoadEvolution("DFTB+/geo_end_novdw.xyz")
    else:
        md.LoadEvolution("DFTB+/geo_end_"+str(vdw)+".xyz")  
    evolution = md.evolution



    dist_x = []
    dist_y = []
    dist_z = []

    forces = []
    start_step = 5000

    for i in range(start_step,len(evolution)):
        forces.append(md.GetForcesOnFrame(i,vdw))
    for j in range(Length):
        v_lower_x = []
        v_lower_y = []
        v_lower_z = []
        for i in range(len(forces)):
            v_lower_x.append(forces[i][j][0])
            v_lower_x.append(forces[i][j+Length][0])
            v_lower_y.append(forces[i][j][1])
            v_lower_y.append(forces[i][j+Length][1])
            v_lower_z.append(forces[i][j][2])
            v_lower_z.append(-(forces[i][j+Length][2]))
        v_lower_x=np.array(v_lower_x)
        v_lower_y=np.array(v_lower_y)
        v_lower_z=np.array(v_lower_z)
        # dist_x.append(np.average(v_lower_x))
        # dist_y.append(np.average(v_lower_y))
        # dist_z.append(np.average(v_lower_z))

        dist_x.append(ss.moment(v_lower_x,moment=2))
        dist_y.append(ss.moment(v_lower_y,moment=2))
        dist_z.append(ss.moment(v_lower_z,moment=2))

        print(np.std(v_lower_x),np.std(v_lower_y),np.std(v_lower_z))

    if vdw == None:
        vdw = "novdw"
    with open('out/force_dist_'+str(vdw)+'.txt', 'w') as f:
        for i in range(len(dist_x)):
            f.write(str(i)+ " " +str(dist_x[i])+ " " +str(dist_y[i])+ " " +str(dist_z[i])+"\n")
