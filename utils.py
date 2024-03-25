import numpy as np
import numpy.linalg as LA
from mdhandler import Handler as MDH
import structures
import collections.abc
import math as m

def StaticOverEvolve(temp):
    Nat=10
    R0=2.4
    Chain1 = structures.Chain(Nat,R0)
    Chain1.SaveGeometry()
    Chain1.RunOptimize()
    Chain1.LoadGeometry()

    Chain2 = structures.Chain(Nat,R0)
    Chain2.SaveGeometry()
    Chain2.RunOptimize()
    Chain2.LoadGeometry()

    Chain2.MoveAll([0,0,75])

    Chain1.add(Chain2)

    md = MDH(Chain1,False)
    md.RunMD(5000,temp,[0,9,10,19])
    ForcesMBD = md.GetForcesSOE("MBD")
    ForcesPW = md.GetForcesSOE("PW")
    ForcesShort = md.GetForcesSOE()


    with open('out/SOE_'+str(temp)+'.txt', 'w') as f:
        for i in range(len(ForcesShort)):
            F_lower_MBD = 0
            F_lower_PW = 0
            F_lower_Short = 0
            for k in range(10):
                F_lower_MBD = F_lower_MBD + ForcesMBD[i][k][2]
                F_lower_PW = F_lower_PW + ForcesPW[i][k][2]
                F_lower_Short = F_lower_Short + ForcesShort[i][k][2]
            F_upper_MBD = 0
            F_upper_PW = 0
            F_upper_Short = 0
            for k in range(10,20):
                F_upper_MBD = F_upper_MBD + ForcesMBD[i][k][2]
                F_upper_PW = F_upper_PW + ForcesPW[i][k][2]
                F_upper_Short = F_upper_Short + ForcesShort[i][k][2]
            f.write(str(i)+ " " +str(F_upper_MBD-F_upper_Short)+ " "+ str(F_lower_MBD-F_lower_Short)+ " " +str(F_upper_PW-F_upper_Short)+ " "+ str(F_lower_PW-F_lower_Short)+ " " +str(F_upper_Short)+ " "+ str(F_lower_Short)+"\n")

def CorrelationOverEvolve(temp):
    Nat=10
    R0=2.4
    Chain1 = structures.Chain(Nat,R0)
    Chain1.SaveGeometry()
    Chain1.RunOptimize()
    Chain1.LoadGeometry()

    Chain2 = structures.Chain(Nat,R0)
    Chain2.SaveGeometry()
    Chain2.RunOptimize()
    Chain2.LoadGeometry()

    Chain2.MoveAll([0,0,20])
    Chain1.add(Chain2)

    Navg=100
    corrxx = np.zeros(10)
    corryy = np.zeros(10)
    corrzz = np.zeros(10)

    corrxy = np.zeros(10)
    corrxz = np.zeros(10)
    corryz = np.zeros(10)

    corrxx_abs = np.zeros(10)
    corryy_abs = np.zeros(10)
    corrzz_abs = np.zeros(10)

    corrxy_abs = np.zeros(10)
    corrxz_abs = np.zeros(10)
    corryz_abs = np.zeros(10)

    for k in range(Navg):
        print(str(k)+"/"+str(Navg))
        md = MDH(Chain1,False)
        md.RunMD(5000,temp,[0,9,10,19])
        md.LoadEvolution()
        evolution = md.evolution
        for j in range(10):
            v_lower_x = []
            v_lower_y = []
            v_lower_z = []
            v_lower_x_abs = []
            v_lower_y_abs = []
            v_lower_z_abs = []
            for i in range(len(evolution)):
                v_lower_x.append(evolution[i][j][3])
                v_lower_y.append(evolution[i][j][4])
                v_lower_z.append(evolution[i][j][5])
                v_lower_x_abs.append(np.abs(evolution[i][j][3]))
                v_lower_y_abs.append(np.abs(evolution[i][j][4]))
                v_lower_z_abs.append(np.abs(evolution[i][j][5]))
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
                v_upper_x.append(evolution[i][j+10][3])
                v_upper_y.append(evolution[i][j+10][4])
                v_upper_z.append(evolution[i][j+10][5])
                v_upper_x_abs.append(np.abs(evolution[i][j+10][3]))
                v_upper_y_abs.append(np.abs(evolution[i][j+10][4]))
                v_upper_z_abs.append(np.abs(evolution[i][j+10][5]))
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


def _write_file(name,content):
    if isinstance(content[0], collections.abc.Sized) == True:
        with open('out/'+name, 'w') as f:
            for i in range(len(content)):
                for j in range(len(content[i])):
                    f.write(str(content[i][j])+' ')
                f.write('\n')
    else:
        with open('out/'+name, 'w') as f:
            for i in range(len(content)):
                f.write(str(i)+' '+str(content[i])+' '+'\n')

def cart2sph(coord_cart):
    x = coord_cart[0]
    y = coord_cart[1]
    z = coord_cart[2]
    XsqPlusYsq = x**2 + y**2
    r = m.sqrt(XsqPlusYsq + z**2)               # r
    elev = m.atan2(z,m.sqrt(XsqPlusYsq))+np.pi/2     # theta
    az = m.atan2(y,x)                           # phi
    return [r, elev, az]

def cart2sph_list(coord_cart):
    coord_spher = []
    for i in range(len(coord_cart)):
        aux = cart2sph(coord_cart[i])
        coord_spher.append(aux)
    return coord_spher