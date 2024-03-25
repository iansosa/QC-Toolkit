import numpy as np
import numpy.linalg as LA
from mdhandler import Handler as MDH
from bondcalc import Bonds
import structures
import sys
import vdw
import functions
from scipy.optimize import minimize 
from heapq import nsmallest
from operator import itemgetter
import random
import utils
import copy 

def measuredecay(structure,r,N_theta,N_phi,chain_length):
    N=1
    R0=1

    init_pos = []
    for i in range(chain_length):
        init_pos.append([i,0,0])
    probing = structures.Custom(init_pos)
    pos = structure.PosAsList()

    x_med = 0
    y_med = 0
    z_med = 0
    for i in range(len(pos)):
        x_med = x_med + pos[i][0]
        y_med = y_med + pos[i][1]
        z_med = z_med + pos[i][2]
    x_med = x_med/len(pos)
    y_med = y_med/len(pos)
    z_med = z_med/len(pos)



    structure.MoveAll([-x_med,-y_med,-z_med])

    probing.add(structure)
    Dtheta=np.pi/N_theta
    Dphi=2*np.pi/N_phi


    theta = 0
    phi = -np.pi

    mbdrates = []
    pwrates = []
    for i in range(N_theta+1):
        for j in range(N_phi+1):
            mbddecay, pwdecay, quotient= probedecay(probing,r,theta,phi,0.1,chain_length)
            with open('out/decay_'+str(r)+'.txt', 'a') as f:
                f.write(str(r)+' '+str(theta)+' '+str(phi)+' '+str(mbddecay)+' '+str(pwdecay)+' '+str(quotient)+'\n')
            phi = phi + Dphi

        phi = -np.pi
        theta = theta + Dtheta

def measuredecay_tilt(structure,r,N_theta,N_phi,chain_length):
    R0=2.2676711863
    structure.Center()
    

    Dtheta=np.pi/(2*N_theta)
    Dphi=2*np.pi/N_phi
    theta = 0
    phi = -np.pi

    mbdrates = []
    pwrates = []
    for i in range(N_theta+1):
        for j in range(N_phi+1):
            r_aux=0
            init_pos = []
            for i in range(chain_length):
                init_pos.append([r_aux*np.cos(phi)*np.sin(theta),r_aux*np.sin(phi)*np.sin(theta),r_aux*np.cos(theta)])
                r_aux = r_aux + R0
            probing = structures.Custom(init_pos)
            probing.MoveAll([0,0,r])
            probing.add(structure)

            mbddecay, pwdecay, quotient= probedecay_tilt(probing,r,0.1,chain_length)
            with open('out/decay_tilt_'+str(r)+'.txt', 'a') as f:
                f.write(str(r)+' '+str(theta)+' '+str(phi)+' '+str(mbddecay)+' '+str(pwdecay)+' '+str(quotient)+'\n')
            phi = phi + Dphi

        phi = -np.pi
        theta = theta + Dtheta

def probedecay_tilt(structure,r,dr,chain_length): #returns the decay rate of both PW and MBD for an atom at a distance r and angles theta,phi from the structure. Atom 0 i s the probing atom

    structure.SaveGeometry()
    structure.RunStatic(vdw="MBD",read_charges=False)
    FMBD = structure.GetForces()
    structure.RunStatic(vdw="TS",read_charges=True)
    FPW = structure.GetForces()


    FMBD_start = 0
    FPW_start = 0
    for i in range(chain_length):
        FMBD_start = FMBD_start + np.inner(FMBD[i],np.array([0,0,1]))
        FPW_start = FPW_start + np.inner(FPW[i],np.array([0,0,1]))

    print("MBD PW start "+str(FMBD_start)+" "+str(FPW_start))

    for i in range(chain_length):
        structure.Displace(i,[0,0,dr])


    structure.SaveGeometry()
    structure.RunStatic(vdw="MBD",read_charges=True)
    FMBD = structure.GetForces()
    structure.RunStatic(vdw="TS",read_charges=True)
    FPW = structure.GetForces()


    FMBD_end = 0
    FPW_end = 0
    for i in range(chain_length):
        FMBD_end = FMBD_end + np.inner(FMBD[i],np.array([0,0,1]))
        FPW_end = FPW_end + np.inner(FPW[i],np.array([0,0,1]))

    print("MBD PW end "+str(FMBD_end)+" "+str(FPW_end))


    Decay_MBD = (np.log(np.fabs(FMBD_end))-np.log(np.fabs(FMBD_start)))/(np.log(r-6+dr)-np.log(r-6))
    Decay_PW = (np.log(np.fabs(FPW_end))-np.log(np.fabs(FPW_start)))/(np.log(r-6+dr)-np.log(r-6))

    quotient = Decay_MBD/Decay_PW


    return Decay_MBD, Decay_PW, quotient

def measuredecay_cilindrical(structure,r,z_bottom,z_top,N_z,N_phi):
    N=1
    R0=1
    probing = structures.Custom([[0,0,0],[1,0,0],[2,0,0],[3,0,0],[4,0,0],[5,0,0],[6,0,0],[7,0,0],[8,0,0],[9,0,0]])
    pos = structure.PosAsList()
    x_med = 0
    y_med = 0
    z_med = 0
    for i in range(len(pos)):
        x_med = x_med + pos[i][0]
        y_med = y_med + pos[i][1]
        z_med = z_med + pos[i][2]
    x_med = x_med/len(pos)
    y_med = y_med/len(pos)
    z_med = z_med/len(pos)

    structure.MoveAll([-x_med,-y_med,-z_med])

    probing.add(structure)
    Dz=(z_top-z_bottom)/N_z
    Dphi=2*np.pi/N_phi


    z = z_bottom
    phi = -np.pi

    mbdrates = []
    pwrates = []
    for i in range(N_z+1):
        for j in range(N_phi+1):
            mbddecay, pwdecay, quotient= probedecay_cilidricals(probing,r,z,phi,0.1)
            with open('out/decay_'+str(r)+'.txt', 'a') as f:
                f.write(str(r)+' '+str(z)+' '+str(phi)+' '+str(mbddecay)+' '+str(pwdecay)+' '+str(quotient)+'\n')
            phi = phi + Dphi

        phi = -np.pi
        z = z + Dz

def measuredecay_plane(structure,r,x_bottom,x_top,N_x,y_bottom,y_top,N_y):
    N=1
    R0=1
    probing = structures.Custom([[0,0,0],[1,0,0],[2,0,0],[3,0,0],[4,0,0],[5,0,0],[6,0,0],[7,0,0],[8,0,0],[9,0,0]])
    pos = structure.PosAsList()
    x_med = 0
    y_med = 0
    z_med = 0
    for i in range(len(pos)):
        x_med = x_med + pos[i][0]
        y_med = y_med + pos[i][1]
        z_med = z_med + pos[i][2]
    x_med = x_med/len(pos)
    y_med = y_med/len(pos)
    z_med = z_med/len(pos)

    structure.MoveAll([-x_med,-y_med,-z_med])

    probing.add(structure)
    Dx=(x_top-x_bottom)/N_x
    Dy=(y_top-y_bottom)/N_y


    x = x_bottom
    y = y_bottom

    mbdrates = []
    pwrates = []
    for i in range(N_x+1):
        for j in range(N_y+1):
            mbddecay, pwdecay, quotient= probedecay_plane(probing,r,x,y,0.1)
            with open('out/decay_'+str(r)+'.txt', 'a') as f:
                f.write(str(r)+' '+str(x)+' '+str(y)+' '+str(mbddecay)+' '+str(pwdecay)+' '+str(quotient)+'\n')
            y = y + Dy
        y = y_bottom
        x = x + Dx


def probedecay(structure,r,theta,phi,dr,chain_length): #returns the decay rate of both PW and MBD for an atom at a distance r and angles theta,phi from the structure. Atom 0 i s the probing atom

    R0=2.2676711863
    mbd = vdw.vdWclass()

    rv = []
    rv.append(r)
    for i in range(chain_length-1):
        rv.append(rv[i]+R0)

    r1=r+R0
    r2=r1+R0
    r3=r2+R0
    r4=r3+R0
    r5=r4+R0
    r6=r5+R0
    r7=r6+R0
    r8=r7+R0
    r9=r8+R0

    for i in range(chain_length):
        structure.SetAtomPos(i,[rv[i]*np.cos(phi)*np.sin(theta),rv[i]*np.sin(phi)*np.sin(theta),rv[i]*np.cos(theta)])   


    structure.SaveGeometry()
    structure.RunStatic(vdw="MBD",read_charges=False)
    FMBD = structure.GetForces()
    structure.RunStatic(vdw="TS",read_charges=True)
    FPW = structure.GetForces()

    # FMBD = mbd.calculate(structure.PosAsList(),["forces"],"MBD")
    # FPW = mbd.calculate(structure.PosAsList(),["forces"],"TS")
    # FMBD_start = np.inner(FMBD[0],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[1],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[2],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[3],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[4],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))
    # FPW_start = np.inner(FPW[0],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[1],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[2],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[3],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[4],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))

    FMBD_start = 0
    FPW_start = 0
    for i in range(chain_length):
        FMBD_start = FMBD_start + np.inner(FMBD[i],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))
        FPW_start = FPW_start + np.inner(FPW[i],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))
    # FMBD_start = np.inner(FMBD[0],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[1],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[2],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[3],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[4],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[5],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[6],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[7],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[8],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[9],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))
    # FPW_start = np.inner(FPW[0],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[1],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[2],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[3],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[4],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[5],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[6],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[7],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[8],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[9],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))


    print("MBD PW start "+str(FMBD_start)+" "+str(FPW_start))

    for i in range(chain_length):
        structure.Displace(i,[dr*np.cos(phi)*np.sin(theta),dr*np.sin(phi)*np.sin(theta),dr*np.cos(theta)])   



    structure.SaveGeometry()
    structure.RunStatic(vdw="MBD",read_charges=True)
    FMBD = structure.GetForces()
    structure.RunStatic(vdw="TS",read_charges=True)
    FPW = structure.GetForces()

    # FMBD = mbd.calculate(structure.PosAsList(),["forces"],"MBD")
    # FPW = mbd.calculate(structure.PosAsList(),["forces"],"TS")
    # FMBD_end = np.inner(FMBD[0],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[1],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[2],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[3],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[4],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))
    # FPW_end = np.inner(FPW[0],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[1],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[2],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[3],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[4],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))

    FMBD_end = 0
    FPW_end = 0
    for i in range(chain_length):
        FMBD_end = FMBD_end + np.inner(FMBD[i],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))
        FPW_end = FPW_end + np.inner(FPW[i],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))
    # FMBD_end = np.inner(FMBD[0],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[1],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[2],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[3],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[4],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[5],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[6],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[7],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[8],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[9],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))
    # FPW_end= np.inner(FPW[0],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[1],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[2],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[3],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[4],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[5],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[6],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[7],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[8],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[9],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))


    print("MBD PW end "+str(FMBD_end)+" "+str(FPW_end))


    Decay_MBD = (np.log(np.fabs(FMBD_end))-np.log(np.fabs(FMBD_start)))/(np.log(r-6+dr)-np.log(r-6))
    Decay_PW = (np.log(np.fabs(FPW_end))-np.log(np.fabs(FPW_start)))/(np.log(r-6+dr)-np.log(r-6))
    # Decay_MBD = LA.norm(FMBD_end-FMBD_start)/dr
    # Decay_PW = LA.norm(FPW_end-FPW_start)/dr
    quotient = Decay_MBD/Decay_PW


    return Decay_MBD, Decay_PW, quotient

def probedecay_cilidricals(structure,r,z,phi,dr): #returns the decay rate of both PW and MBD for an atom at a distance r and angles theta,phi from the structure. Atom 0 i s the probing atom

    R0=2.4566437851
    mbd = vdw.vdWclass()

    r1=r+2.4221560179
    r2=r1+2.4952981429
    r3=r2+2.4276410870685937
    r4=r3+2.4739789030775247
    r5=r4+2.4322521148568027
    r6=r5+2.4739788804
    r7=r6+2.4276410606124310
    r8=r7+2.4952981170487314
    r9=r8+2.4221559967599745

    print(z)
    structure.SetAtomPos(0,[r*np.cos(phi),r*np.sin(phi),z])
    structure.SetAtomPos(1,[r1*np.cos(phi),r1*np.sin(phi),z])
    structure.SetAtomPos(2,[r2*np.cos(phi),r2*np.sin(phi),z])
    structure.SetAtomPos(3,[r3*np.cos(phi),r3*np.sin(phi),z])
    structure.SetAtomPos(4,[r4*np.cos(phi),r4*np.sin(phi),z])
    structure.SetAtomPos(5,[r5*np.cos(phi),r5*np.sin(phi),z])
    structure.SetAtomPos(6,[r6*np.cos(phi),r6*np.sin(phi),z])
    structure.SetAtomPos(7,[r7*np.cos(phi),r7*np.sin(phi),z])
    structure.SetAtomPos(8,[r8*np.cos(phi),r8*np.sin(phi),z])
    structure.SetAtomPos(9,[r9*np.cos(phi),r9*np.sin(phi),z])

    structure.SaveGeometry()
    structure.RunStatic(vdw="MBD",read_charges=False)
    FMBD = structure.GetForces()
    structure.RunStatic(vdw="TS",read_charges=True)
    FPW = structure.GetForces()

    # FMBD = mbd.calculate(structure.PosAsList(),["forces"],"MBD")
    # FPW = mbd.calculate(structure.PosAsList(),["forces"],"TS")


    # FMBD_start = np.inner(FMBD[0],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[1],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[2],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[3],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[4],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))
    # FPW_start = np.inner(FPW[0],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[1],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[2],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[3],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[4],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))
    FMBD_start = np.inner(FMBD[0],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FMBD[1],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FMBD[2],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FMBD[3],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FMBD[4],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FMBD[5],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FMBD[6],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FMBD[7],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FMBD[8],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FMBD[9],np.array([np.cos(phi),np.sin(phi),0]))
    FPW_start = np.inner(FPW[0],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FPW[1],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FPW[2],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FPW[3],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FPW[4],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FPW[5],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FPW[6],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FPW[7],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FPW[8],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FPW[9],np.array([np.cos(phi),np.sin(phi),0]))

    # structure.ShowStruct()

    print("MBD PW start "+str(FMBD_start)+" "+str(FPW_start))


    structure.Displace(0,[dr*np.cos(phi),dr*np.sin(phi),0])
    structure.Displace(1,[dr*np.cos(phi),dr*np.sin(phi),0])
    structure.Displace(2,[dr*np.cos(phi),dr*np.sin(phi),0])
    structure.Displace(3,[dr*np.cos(phi),dr*np.sin(phi),0])
    structure.Displace(4,[dr*np.cos(phi),dr*np.sin(phi),0])
    structure.Displace(5,[dr*np.cos(phi),dr*np.sin(phi),0])
    structure.Displace(6,[dr*np.cos(phi),dr*np.sin(phi),0])
    structure.Displace(7,[dr*np.cos(phi),dr*np.sin(phi),0])
    structure.Displace(8,[dr*np.cos(phi),dr*np.sin(phi),0])
    structure.Displace(9,[dr*np.cos(phi),dr*np.sin(phi),0])

    structure.SaveGeometry()
    structure.RunStatic(vdw="MBD",read_charges=True)
    FMBD = structure.GetForces()
    structure.RunStatic(vdw="TS",read_charges=True)
    FPW = structure.GetForces()


    # FMBD = mbd.calculate(structure.PosAsList(),["forces"],"MBD")
    # FPW = mbd.calculate(structure.PosAsList(),["forces"],"TS")

    # FMBD_end = np.inner(FMBD[0],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[1],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[2],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[3],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[4],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))
    # FPW_end = np.inner(FPW[0],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[1],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[2],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[3],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[4],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))
    FMBD_end = np.inner(FMBD[0],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FMBD[1],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FMBD[2],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FMBD[3],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FMBD[4],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FMBD[5],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FMBD[6],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FMBD[7],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FMBD[8],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FMBD[9],np.array([np.cos(phi),np.sin(phi),0]))
    FPW_end = np.inner(FPW[0],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FPW[1],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FPW[2],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FPW[3],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FPW[4],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FPW[5],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FPW[6],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FPW[7],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FPW[8],np.array([np.cos(phi),np.sin(phi),0]))+np.inner(FPW[9],np.array([np.cos(phi),np.sin(phi),0]))


    print("MBD PW end "+str(FMBD_end)+" "+str(FPW_end))


    if np.log(np.fabs(FMBD_end))-np.log(np.fabs(FMBD_start)) > 0:
        print("postitive decay MBD")
    else:
        print("negative decay MBD")
    if np.log(np.fabs(FPW_end))-np.log(np.fabs(FPW_start)) > 0:
        print("postitive decay TS")
    else:
        print("negative decay TS")

    Decay_MBD = (np.log(np.fabs(FMBD_end))-np.log(np.fabs(FMBD_start)))/(np.log(r+dr)-np.log(r))
    Decay_PW = (np.log(np.fabs(FPW_end))-np.log(np.fabs(FPW_start)))/(np.log(r+dr)-np.log(r))
    # Decay_MBD = LA.norm(FMBD_end-FMBD_start)/dr
    # Decay_PW = LA.norm(FPW_end-FPW_start)/dr
    quotient = Decay_MBD/Decay_PW


    return Decay_MBD, Decay_PW, quotient

def probedecay_plane(structure,r,x,y,dr): #returns the decay rate of both PW and MBD for an atom at a distance r and angles theta,phi from the structure. Atom 0 i s the probing atom

    R0=2.4566437851
    mbd = vdw.vdWclass()

    r1=r+2.4221560179
    r2=r1+2.4952981429
    r3=r2+2.4276410870685937
    r4=r3+2.4739789030775247
    r5=r4+2.4322521148568027
    r6=r5+2.4739788804
    r7=r6+2.4276410606124310
    r8=r7+2.4952981170487314
    r9=r8+2.4221559967599745


    structure.SetAtomPos(0,[x,y,r])
    structure.SetAtomPos(1,[x,y,r1])
    structure.SetAtomPos(2,[x,y,r2])
    structure.SetAtomPos(3,[x,y,r3])
    structure.SetAtomPos(4,[x,y,r4])
    structure.SetAtomPos(5,[x,y,r5])
    structure.SetAtomPos(6,[x,y,r6])
    structure.SetAtomPos(7,[x,y,r7])
    structure.SetAtomPos(8,[x,y,r8])
    structure.SetAtomPos(9,[x,y,r9])

    structure.SaveGeometry()
    structure.RunStatic(vdw="MBD",read_charges=False)
    FMBD = structure.GetForces()
    structure.RunStatic(vdw="TS",read_charges=True)
    FPW = structure.GetForces()

    # FMBD = mbd.calculate(structure.PosAsList(),["forces"],"MBD")
    # FPW = mbd.calculate(structure.PosAsList(),["forces"],"TS")


    # FMBD_start = np.inner(FMBD[0],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[1],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[2],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[3],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[4],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))
    # FPW_start = np.inner(FPW[0],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[1],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[2],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[3],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[4],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))
    FMBD_start = np.inner(FMBD[0],np.array([0,0,1]))+np.inner(FMBD[1],np.array([0,0,1]))+np.inner(FMBD[2],np.array([0,0,1]))+np.inner(FMBD[3],np.array([0,0,1]))+np.inner(FMBD[4],np.array([0,0,1]))+np.inner(FMBD[5],np.array([0,0,1]))+np.inner(FMBD[6],np.array([0,0,1]))+np.inner(FMBD[7],np.array([0,0,1]))+np.inner(FMBD[8],np.array([0,0,1]))+np.inner(FMBD[9],np.array([0,0,1]))
    FPW_start = np.inner(FPW[0],np.array([0,0,1]))+np.inner(FPW[1],np.array([0,0,1]))+np.inner(FPW[2],np.array([0,0,1]))+np.inner(FPW[3],np.array([0,0,1]))+np.inner(FPW[4],np.array([0,0,1]))+np.inner(FPW[5],np.array([0,0,1]))+np.inner(FPW[6],np.array([0,0,1]))+np.inner(FPW[7],np.array([0,0,1]))+np.inner(FPW[8],np.array([0,0,1]))+np.inner(FPW[9],np.array([0,0,1]))

    # structure.ShowStruct()

    print("MBD PW start "+str(FMBD_start)+" "+str(FPW_start))


    structure.Displace(0,[0,0,dr])
    structure.Displace(1,[0,0,dr])
    structure.Displace(2,[0,0,dr])
    structure.Displace(3,[0,0,dr])
    structure.Displace(4,[0,0,dr])
    structure.Displace(5,[0,0,dr])
    structure.Displace(6,[0,0,dr])
    structure.Displace(7,[0,0,dr])
    structure.Displace(8,[0,0,dr])
    structure.Displace(9,[0,0,dr])

    structure.SaveGeometry()
    structure.RunStatic(vdw="MBD",read_charges=True)
    FMBD = structure.GetForces()
    structure.RunStatic(vdw="TS",read_charges=True)
    FPW = structure.GetForces()


    # FMBD = mbd.calculate(structure.PosAsList(),["forces"],"MBD")
    # FPW = mbd.calculate(structure.PosAsList(),["forces"],"TS")

    # FMBD_end = np.inner(FMBD[0],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[1],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[2],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[3],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FMBD[4],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))
    # FPW_end = np.inner(FPW[0],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[1],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[2],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[3],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))+np.inner(FPW[4],np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))
    FMBD_end = np.inner(FMBD[0],np.array([0,0,1]))+np.inner(FMBD[1],np.array([0,0,1]))+np.inner(FMBD[2],np.array([0,0,1]))+np.inner(FMBD[3],np.array([0,0,1]))+np.inner(FMBD[4],np.array([0,0,1]))+np.inner(FMBD[5],np.array([0,0,1]))+np.inner(FMBD[6],np.array([0,0,1]))+np.inner(FMBD[7],np.array([0,0,1]))+np.inner(FMBD[8],np.array([0,0,1]))+np.inner(FMBD[9],np.array([0,0,1]))
    FPW_end = np.inner(FPW[0],np.array([0,0,1]))+np.inner(FPW[1],np.array([0,0,1]))+np.inner(FPW[2],np.array([0,0,1]))+np.inner(FPW[3],np.array([0,0,1]))+np.inner(FPW[4],np.array([0,0,1]))+np.inner(FPW[5],np.array([0,0,1]))+np.inner(FPW[6],np.array([0,0,1]))+np.inner(FPW[7],np.array([0,0,1]))+np.inner(FPW[8],np.array([0,0,1]))+np.inner(FPW[9],np.array([0,0,1]))


    print("MBD PW end "+str(FMBD_end)+" "+str(FPW_end))


    Decay_MBD = (np.log(np.fabs(FMBD_end))-np.log(np.fabs(FMBD_start)))/(np.log(r+dr)-np.log(r))
    Decay_PW = (np.log(np.fabs(FPW_end))-np.log(np.fabs(FPW_start)))/(np.log(r+dr)-np.log(r))
    # Decay_MBD = LA.norm(FMBD_end-FMBD_start)/dr
    # Decay_PW = LA.norm(FPW_end-FPW_start)/dr
    quotient = Decay_MBD/Decay_PW


    return Decay_MBD, Decay_PW, quotient

def delta_measure():
    geom = structures.FromFile("fullerenes/C180.gen")
    mbd = vdw.vdWclass()
    FMBD = mbd.calculate(geom.PosAsList(),["forces"],"MBD")
    FPW = mbd.calculate(geom.PosAsList(),["forces"],"TS",discrete=True)
    idx_range = list(range(0, 180))
    random.shuffle(idx_range)
    for i in idx_range[:10]:
        min_fun = 10000
        min_x = 10000
        min_bound = 1
        for j in range(100):
            x = minimize(functions.min_delta, [j,3.2125341806],bounds=((-20,10.0),(0,40)),options={'ftol': 1e-8,'maxiter':10000}, args=(FMBD[i],FPW[i],geom.Distances(i)))
            if min_fun > x['fun']:
                min_fun = x['fun']
                min_x = x['x'][0]
                min_bound = x['x'][1]
#        print(i,functions.min_delta([0.2,3.2125341806],FMBD[i],FPW[i],geom.Distances(i)))
        print(min_fun,min_x,min_bound,i,functions.min_delta([0,3.2125341806],FMBD[i],FPW[i],geom.Distances(i)),LA.norm(FMBD[i]))

def measure_min_on(file,idx,d,from_idx):
    if 'Polyethylene/Greek' in file:
        geom_aux = structures.FromFile(file)
        geom_aux.Fold()
        geom_aux.Expand([2,2,2],trim = 1.2)
        a = geom_aux.Distances(idx)
        idx_aux, _ = zip(*nsmallest(1000, enumerate(a), key=itemgetter(1)))
        geom = structures.Custom(geom_aux.PosAsListIdx(idx_aux),types=geom_aux.TypesAsListIdx(idx_aux))
    else:
        geom = structures.FromFile(file)
    mbd = vdw.vdWclass()
    FMBD = mbd.calculate(geom.PosAsList(),["forces"],"MBD", atom_types = geom.types)
    FPW = mbd.calculate(geom.PosAsList(),["forces"],"TS",discrete=True, atom_types = geom.types)
    distances = geom.Distances(0)
    x = minimize(functions.min_delta_salted,d,bounds=((0,10.0),(0,10),(-10,15)),options={'ftol': 1e-8,'maxiter':10000}, args=(FMBD[0],FPW[0],distances))
    d[0]=0
    d[2]=0
    ref = functions.min_delta(d,FMBD[0],FPW[0],distances)
    min_fun = functions.min_delta([x['x'][0],x['x'][1],x['x'][2]],FMBD[0],FPW[0],distances)
    min_x1 = x['x'][0]
    min_x2 = x['x'][1]
    min_x3 = x['x'][2]


    FPW_corr = functions.min_delta_force([x['x'][0],x['x'][1],x['x'][2]],FMBD[0],FPW[0],distances)
    FPW = mbd.calculate(geom.PosAsList(),["forces"],"TS",discrete=False, atom_types = geom.types)
    with open('out/forces_'+str(from_idx)+'.txt', 'a') as f:
        f.write(str(idx)+' '+str(FMBD[0][0])+' '+str(FMBD[0][1])+' '+str(FMBD[0][2])+' '+str(FPW[0][0])+' '+str(FPW[0][1])+' '+str(FPW[0][2])+' '+str(FPW_corr[0])+' '+str(FPW_corr[1])+' '+str(FPW_corr[2])+'\n')

    print(ref,min_fun,min_x1,min_x2,min_x3)
    return [LA.norm(FMBD[0]),ref,min_fun,min_x1,min_x2,min_x3]

def measure_min_on_random_struct(idx,d,from_idx):
    pos = []
    pos.append([0,0,0])
    random.seed(idx)
    while len(pos) < 1000:
        isilegal = False
        # aux = [random.uniform(-50, 50),random.uniform(-50, 50),random.uniform(-50, 50)]
        aux = [random.gauss(pos[-1][0], 5),random.gauss(pos[-1][1], 5),random.gauss(pos[-1][2], 5)]
        if LA.norm(np.array(aux)) < 50:
            for i in range(len(pos)):
                if LA.norm(np.array(aux)-np.array(pos[i])) < 3.5:
                    isilegal = True
            if isilegal == False:
                pos.append(aux)
    geom = structures.Custom(pos)
    geom.SaveGeometry()
    
    mbd = vdw.vdWclass()
    FMBD = mbd.calculate(geom.PosAsList(),["forces"],"MBD", atom_types = geom.types)
    FPW = mbd.calculate(geom.PosAsList(),["forces"],"TS",discrete=True, atom_types = geom.types)
    distances = geom.Distances(0)
    x = minimize(functions.min_delta_salted,d,bounds=((0,10.0),(0,10),(-10,15)),options={'ftol': 1e-8,'maxiter':10000}, args=(FMBD[0],FPW[0],distances))
    d[0]=0
    d[2]=0
    ref = functions.min_delta(d,FMBD[0],FPW[0],distances)
    min_fun = functions.min_delta([x['x'][0],x['x'][1],x['x'][2]],FMBD[0],FPW[0],distances)
    min_x1 = x['x'][0]
    min_x2 = x['x'][1]
    min_x3 = x['x'][2]


    FPW_corr = functions.min_delta_force([x['x'][0],x['x'][1],x['x'][2]],FMBD[0],FPW[0],distances)
    FPW = mbd.calculate(geom.PosAsList(),["forces"],"TS",discrete=False, atom_types = geom.types)
    with open('out/forces_'+str(from_idx)+'.txt', 'a') as f:
        f.write(str(idx)+' '+str(FMBD[0][0])+' '+str(FMBD[0][1])+' '+str(FMBD[0][2])+' '+str(FPW[0][0])+' '+str(FPW[0][1])+' '+str(FPW[0][2])+' '+str(FPW_corr[0])+' '+str(FPW_corr[1])+' '+str(FPW_corr[2])+'\n')

    print(ref,min_fun,min_x1,min_x2,min_x3)
    return [LA.norm(FMBD[0]),ref,min_fun,min_x1,min_x2,min_x3]

def delta_map(file,idx,dd,ddmin,ddmax,dr,drmin,drmax):
    if 'Polyethylene/Greek' in file:
        geom_aux = structures.FromFile(file)
        geom_aux.Fold()
        geom_aux.Expand([2,2,2],trim = 1.2)
        a = geom_aux.Distances(idx)
        idx_aux, _ = zip(*nsmallest(1000, enumerate(a), key=itemgetter(1)))
        geom = structures.Custom(geom_aux.PosAsListIdx(idx_aux),types=geom_aux.TypesAsListIdx(idx_aux))
    else:
        geom = structures.FromFile(file)
    mbd = vdw.vdWclass()
    FMBD = mbd.calculate(geom.PosAsList(),["forces"],"MBD")
    FPW = mbd.calculate(geom.PosAsList(),["forces"],"TS",discrete=True)
    min_fun = 10000
    min_x = 10000
    min_bound = 1
    ND=int((ddmax-ddmin)/dd)
    NR=int((drmax-drmin)/dr)
    distances = geom.Distances(idx)
    for i in range(ND):
        for j in range(NR):
            res = functions.min_delta([dd*i+ddmin,dr*j+drmin],FMBD[idx],FPW[idx],distances)
            print(dd*i+ddmin,dr*j+drmin,res)
            with open('out/delta_map_'+str(idx)+'.txt', 'a') as f:
                f.write(str(dd*i+ddmin)+' '+str(dr*j+drmin)+' '+str(res)+'\n')

def delta_calc_min(file,idx,dd,ddmin,ddmax,dr,drmin,drmax,dp,dpmin,dpmax):
    if 'Polyethylene/Greek' in file:
        geom_aux = structures.FromFile(file)
        geom_aux.Fold()
        geom_aux.Expand([2,2,2],trim = 1.2)
        a = geom_aux.Distances(idx)
        idx_aux, _ = zip(*nsmallest(1000, enumerate(a), key=itemgetter(1)))
        geom = structures.Custom(geom_aux.PosAsListIdx(idx_aux),types=geom_aux.TypesAsListIdx(idx_aux))
    else:
        geom = structures.FromFile(file)
    mbd = vdw.vdWclass()
    FMBD = mbd.calculate(geom.PosAsList(),["forces"],"MBD",atom_types = geom.types)
    FPW = mbd.calculate(geom.PosAsList(),["forces"],"TS",discrete=True,atom_types = geom.types)
    min_fun = 10000
    min_x = 10000
    min_bound = 1
    ND=int((ddmax-ddmin)/dd)
    NR=int((drmax-drmin)/dr)
    NP=int((dpmax-dpmin)/dp)
    distances = geom.Distances(idx)
    res_prev = 10000
    d_prev = 0
    r_prev = 0
    p_prev = 0
    count = 0
    for i in range(ND):
        for j in range(NR):
            for k in range(NP):
                res = functions.min_delta([dd*i+ddmin,dr*j+drmin,dp*k+dpmin],FMBD[idx],FPW[idx],distances)
                count = count + 1
                if res_prev > res:
                    res_prev = res
                    d_prev = dd*i+ddmin
                    r_prev = dr*j+drmin
                    p_prev = dp*k+dpmin
                print(d_prev,r_prev,p_prev,res_prev, 100*(count/(ND*NR*NP)))

def delta_stats(from_idx,to_idx):
    for i in range(from_idx,to_idx):
        # r = measure_min_on("Polyethylene/Greek/confin100000.gro",idx=i,d=[3,4,-8],from_idx=from_idx)
        r = measure_min_on_random_struct(idx=i,d=[3,4,-8],from_idx=from_idx)
        with open('out/fitting_'+str(from_idx)+'.txt', 'a') as f:
            f.write(str(i)+' '+str(r[0])+' '+str(r[1])+' '+str(r[2])+' '+str(r[3])+' '+str(r[4])+' '+str(r[5])+'\n')

def platonic_solid_randomwalk(file,num_stats = 100, min_distance = 3.5):


    magnitude = 0.1
    num_steps = 1000

    stats = []
    for l in range(num_stats):
        sample = []
        random.seed(l)
        geom = structures.FromFile("Polyhedra/"+file)
        N_at_poly = geom.Nat
        # geom_aux = structures.Custom([[0,0,0],[2.5/np.sqrt(2),2.5/np.sqrt(2),0],[5/np.sqrt(2),5/np.sqrt(2),0]])
        geom_aux = structures.Custom([[-3.5/np.sqrt(2),-3.5/np.sqrt(2),0],[0,0,0],[3.5/np.sqrt(2),3.5/np.sqrt(2),0]])
        # geom_aux = structures.Custom([[0,0,-3.5],[0,0,0],[0,0,3.5]])
        # geom_aux = structures.Custom([[0,0,0]])
        Nat_aux = geom_aux.Nat
        geom.add(geom_aux)

        for k in range(num_steps):
            for i in range(N_at_poly):
                success = False
                x_aux = geom.x[i]
                y_aux = geom.y[i]
                z_aux = geom.z[i] 
                tries = 0
                while success == False:
                    success = True
                    geom.x[i] = x_aux
                    geom.y[i] = y_aux
                    geom.z[i] = z_aux
                    in_radious = LA.norm(np.array([geom.x[i],geom.y[i],geom.z[i]]))
                    # geom.Displace(i,[random.uniform(-magnitude, magnitude),random.uniform(-magnitude, magnitude),random.uniform(-magnitude,magnitude)])
                    # geom.Displace(i,[random.uniform(-magnitude, magnitude),random.uniform(-magnitude, magnitude),random.uniform(-magnitude,magnitude)+0.05])
                    geom.Displace(i,[random.uniform(-magnitude, magnitude)+0.035355,random.uniform(-magnitude, magnitude)+0.035355,random.uniform(-magnitude,magnitude)])
                    out_radious = LA.norm(np.array([geom.x[i],geom.y[i],geom.z[i]]))
                    geom.x[i] = geom.x[i]*in_radious/out_radious
                    geom.y[i] = geom.y[i]*in_radious/out_radious
                    geom.z[i] = geom.z[i]*in_radious/out_radious
                    distances = geom.Distances(i)
                    for j in range(N_at_poly):
                        if distances[j] < min_distance and i != j:
                            success = False
                            tries = tries + 1
                        if tries == 100:
                            geom.x[i] = x_aux
                            geom.y[i] = y_aux
                            geom.z[i] = z_aux
                            success = True
            mbd = vdw.vdWclass()
            FMBD = mbd.calculate(geom.PosAsList(),["forces"],"MBD",atom_types = geom.types)
            FPW = mbd.calculate(geom.PosAsList(),["forces"],"TS",discrete=False,atom_types = geom.types)
            TFMBD = 0
            TFPW = 0
            for i in range(Nat_aux):
                TFMBD = TFMBD + FMBD[N_at_poly+i]
                TFPW = TFPW + FPW[N_at_poly+i]
            pos = np.array(geom.PosAsList())
            asym_vec = np.array([0,0,0])

            geom_aux = copy.deepcopy(geom)
            to_remove = []
            for i in range(Nat_aux):
                to_remove.append(N_at_poly+i)
            geom_aux.RemoveAtoms(to_remove)
            for i in range(N_at_poly):
                pos[i] = pos[i]/LA.norm(pos[i])
                asym_vec = asym_vec + pos[i]
            asym_vec = asym_vec / (N_at_poly)

            r_inv = 0
            for i in range(N_at_poly):
                aux = geom_aux.Distances(i)
                aux.sort()
                r_inv = r_inv + ( (1.0/aux[1]) + (1.0/aux[2]) + (1.0/aux[3]) + (1.0/aux[4]))/(4.0*0.2857)
            r_inv = r_inv /(N_at_poly)

            order_2 = 0
            for i in range(N_at_poly):
                aux = geom_aux.Distances(i)
                aux.sort()
                dis_avg = (aux[1] + aux[2])/2.0
                dis_rel = (aux[2]-aux[1])/dis_avg
                order_2 = order_2 + 1 - np.sqrt(abs(dis_rel))
            order_2 = order_2 /(N_at_poly)

            order_3 = 0
            for i in range(N_at_poly):
                aux = geom_aux.Distances(i)
                aux.sort()
                dis_avg = (aux[1] + aux[2] + aux[3])/3.0
                dis_rel = (aux[3]-aux[1])/dis_avg
                order_3 = order_3 + 1 - np.sqrt(abs(dis_rel))
            order_3 = order_3 /(N_at_poly)

            order_4 = 0
            for i in range(N_at_poly):
                aux = geom_aux.Distances(i)
                aux.sort()
                dis_avg = (aux[1] + aux[2] + aux[3] + aux[4])/4.0
                dis_rel = (aux[4]-aux[1])/dis_avg
                order_4 = order_4 + 1 - np.sqrt(abs(dis_rel))
            order_4 = order_4 /(N_at_poly)

            order_5 = 0
            for i in range(N_at_poly):
                aux = geom_aux.Distances(i)
                aux.sort()
                dis_avg = (aux[1] + aux[2] + aux[3] + aux[4] + aux[5])/5.0
                dis_rel = (aux[5]-aux[1])/dis_avg
                order_5 = order_5 + 1 - np.sqrt(abs(dis_rel))
            order_5 = order_5 /(N_at_poly)

            properties = np.array([k,LA.norm(asym_vec),r_inv,order_2,order_3,order_4,order_5,r_inv*order_4,LA.norm(TFMBD)/LA.norm(TFPW),np.inner(TFMBD,TFPW)/(LA.norm(TFMBD)*LA.norm(TFPW))])
            sample.append(properties)
        sample = np.array(sample)
        stats.append(np.array(sample))
        geom.SaveGeometry()
        print(l,num_stats)

    stats = np.array(stats)
    # out = np.average(stats,axis=0)
    out = np.std(stats,axis=0)
    name = "Sym_break_"+file[:-4]+"_110_S"+str(num_stats)+"_mdist"+str(min_distance)
    # name = "Sym_break_N"+str(geom.Nat)+"_"+str(num_stats)
    utils._write_file(name+".txt",out)

def Sym_break(file,num_stats = 100,magnitude = 20):

    # magnitude = 0.1

    sample = []
    for l in range(num_stats):
        random.seed(l)
        drift = random.uniform(0,50)
        geom = structures.FromFile("Polyhedra/"+file)
        N_at_poly = geom.Nat
        # geom_aux = structures.Custom([[0,0,0],[2.5/np.sqrt(2),2.5/np.sqrt(2),0],[5/np.sqrt(2),5/np.sqrt(2),0]])
        geom_aux = structures.Custom([[-3.5/np.sqrt(2),-3.5/np.sqrt(2),0],[0,0,0],[3.5/np.sqrt(2),3.5/np.sqrt(2),0]])
        # geom_aux = structures.Custom([[5/np.sqrt(2),5/np.sqrt(2),0]])
        geom.add(geom_aux)
        for i in range(N_at_poly):
            success = False
            x_aux = geom.x[i]
            y_aux = geom.y[i]
            z_aux = geom.z[i] 
            tries = 0
            while success == False:
                success = True
                geom.x[i] = x_aux
                geom.y[i] = y_aux
                geom.z[i] = z_aux
                in_radious = LA.norm(np.array([geom.x[i],geom.y[i],geom.z[i]]))
                geom.Displace(i,[random.uniform(-magnitude, magnitude),random.uniform(-magnitude, magnitude),random.uniform(-magnitude,magnitude)+drift])
                # geom.Displace(i,[random.uniform(-magnitude, magnitude),random.uniform(-magnitude, magnitude),random.uniform(-magnitude,magnitude)+0.05])
                # geom.Displace(i,[random.uniform(-magnitude, magnitude)+0.035355,random.uniform(-magnitude, magnitude)+0.035355,random.uniform(-magnitude,magnitude)])
                out_radious = LA.norm(np.array([geom.x[i],geom.y[i],geom.z[i]]))
                geom.x[i] = geom.x[i]*in_radious/out_radious
                geom.y[i] = geom.y[i]*in_radious/out_radious
                geom.z[i] = geom.z[i]*in_radious/out_radious
                distances = geom.Distances(i)
                for j in range(N_at_poly):
                    if distances[j] < 3.5 and i != j:
                        success = False
                        tries = tries + 1
                    if tries > 5000:
                        geom.x[i] = x_aux
                        geom.y[i] = y_aux
                        geom.z[i] = z_aux
                        success = True
        mbd = vdw.vdWclass()
        FMBD = mbd.calculate(geom.PosAsList(),["forces"],"MBD",atom_types = geom.types)
        FPW = mbd.calculate(geom.PosAsList(),["forces"],"TS",discrete=False,atom_types = geom.types)
        TFMBD = FMBD[N_at_poly]+FMBD[N_at_poly+1]+FMBD[N_at_poly+2]
        TFPW = FPW[N_at_poly]+FPW[N_at_poly+1]+FPW[N_at_poly+2]
        pos = np.array(geom.PosAsList())
        asym_vec = np.array([0,0,0])

        geom_aux = copy.deepcopy(geom)
        geom_aux.RemoveAtoms([N_at_poly,N_at_poly+1,N_at_poly+2])
        for i in range(N_at_poly):
            pos[i] = pos[i]/LA.norm(pos[i])
            asym_vec = asym_vec + pos[i]
        asym_vec = asym_vec / (N_at_poly)

        r_inv = 0
        for i in range(N_at_poly):
            aux = geom_aux.Distances(i)
            aux.sort()
            r_inv = r_inv + ( (1.0/aux[1]) + (1.0/aux[2]) + (1.0/aux[3]) + (1.0/aux[4]))/(4.0*0.2857)
        r_inv = r_inv /(N_at_poly)


        order_2 = 0
        for i in range(N_at_poly):
            aux = geom_aux.Distances(i)
            aux.sort()
            dis_avg = (aux[1] + aux[2])/2.0
            dis_rel = (aux[2]-aux[1])/dis_avg
            order_2 = order_2 + 1 - np.sqrt(abs(dis_rel))
        order_2 = order_2 /(N_at_poly)

        order_3 = 0
        for i in range(N_at_poly):
            aux = geom_aux.Distances(i)
            aux.sort()
            dis_avg = (aux[1] + aux[2] + aux[3])/3.0
            dis_rel = (aux[3]-aux[1])/dis_avg
            order_3 = order_3 + 1 - np.sqrt(abs(dis_rel))
        order_3 = order_3 /(N_at_poly)

        order_4 = 0
        for i in range(N_at_poly):
            aux = geom_aux.Distances(i)
            aux.sort()
            dis_avg = (aux[1] + aux[2] + aux[3] + aux[4])/4.0
            dis_rel = (aux[4]-aux[1])/dis_avg
            order_4 = order_4 + 1 - np.sqrt(abs(dis_rel))
        order_4 = order_4 /(N_at_poly)

        order_5 = 0
        for i in range(N_at_poly):
            aux = geom_aux.Distances(i)
            aux.sort()
            dis_avg = (aux[1] + aux[2] + aux[3] + aux[4] + aux[5])/5.0
            dis_rel = (aux[5]-aux[1])/dis_avg
            order_5 = order_5 + 1 - np.sqrt(abs(dis_rel))
        order_5 = order_5 /(N_at_poly)
        properties = np.array([LA.norm(asym_vec),r_inv,order_2,order_3,order_4,order_5,r_inv*order_4,LA.norm(TFMBD)/LA.norm(TFPW),np.inner(TFMBD,TFPW)/(LA.norm(TFMBD)*LA.norm(TFPW))])
        # print(np.inner(FMBD,FPW)/(LA.norm(FMBD)*LA.norm(FPW)))
        sample.append(properties)
        print(l,num_stats)
        geom.SaveGeometry()

    N_sym_points = 50
    dsym = 1.0/N_sym_points

    dist = np.zeros(N_sym_points)
    dist_rinv = np.zeros(N_sym_points)
    dist_ord_2 = np.zeros(N_sym_points)
    dist_ord_3 = np.zeros(N_sym_points)
    dist_ord_4 = np.zeros(N_sym_points)
    dist_ord_5 = np.zeros(N_sym_points)
    cosine = np.zeros(N_sym_points)
    cummulative_stats = np.zeros(N_sym_points)
    for i in range(len(sample)):
        for k in range(N_sym_points):
            if sample[i][0] > dsym*k and sample[i][0] < dsym*k + dsym:
                dist[k] = dist[k]+sample[i][7]
                dist_ord_2[k] = dist_ord_2[k]+sample[i][2]
                dist_ord_3[k] = dist_ord_3[k]+sample[i][3]
                dist_ord_4[k] = dist_ord_4[k]+sample[i][4]
                dist_ord_5[k] = dist_ord_5[k]+sample[i][5]
                cosine[k] = cosine[k] + sample[i][8]
                dist_rinv[k] = dist_rinv[k]+sample[i][1]
                cummulative_stats[k] = cummulative_stats[k] + 1
    for i in range(N_sym_points):
        if cummulative_stats[i] > 0:
            dist[i] = dist[i] / cummulative_stats[i]
            dist_rinv[i] = dist_rinv[i] / cummulative_stats[i]
            dist_ord_2[i] = dist_ord_2[i] / cummulative_stats[i]
            dist_ord_3[i] = dist_ord_3[i] / cummulative_stats[i]
            dist_ord_4[i] = dist_ord_4[i] / cummulative_stats[i]
            dist_ord_5[i] = dist_ord_5[i] / cummulative_stats[i]
            cosine[i] = cosine[i] / cummulative_stats[i]
        else:
            dist[i] = 0
            dist_rinv[i]= 0
            dist_ord_2[i] = 0
            dist_ord_3[i] = 0
            dist_ord_4[i] = 0
            dist_ord_5[i] = 0
            cosine[i] = 0
    out = []
    for i in range(len(dist)):
        out.append([dsym*i+dsym*0.5,dist[i],dist_rinv[i],dist_ord_2[i],dist_ord_3[i],dist_ord_4[i],dist_ord_5[i],cosine[i],cummulative_stats[i]])

    # name = "Sym_break_"+file[:-4]+"_001_"+str(num_stats)
    name = "Sym_break_N"+str(geom.Nat)+"_"+str(num_stats)
    utils._write_file(name+".txt",out)

def platonic_solid_randomwalk_structdist(file,num_stats = 100, min_distance = 3.5, band = [0.5,0.6]):


    magnitude = 0.1
    num_steps = 1000

    N_theta = 50
    N_phi = 50
    dist = np.zeros((N_theta,N_phi))

    stats = []
    for l in range(num_stats):
        sample = []
        random.seed(l)
        geom = structures.FromFile("Polyhedra/"+file)
        N_at_poly = geom.Nat
        # geom_aux = structures.Custom([[0,0,0],[2.5/np.sqrt(2),2.5/np.sqrt(2),0],[5/np.sqrt(2),5/np.sqrt(2),0]])
        geom_aux = structures.Custom([[-3.5/np.sqrt(2),-3.5/np.sqrt(2),0],[0,0,0],[3.5/np.sqrt(2),3.5/np.sqrt(2),0]])
        # geom_aux = structures.Custom([[0,0,-3.5],[0,0,0],[0,0,3.5]])
        geom.add(geom_aux)


        for k in range(num_steps):
            for i in range(N_at_poly):
                success = False
                x_aux = geom.x[i]
                y_aux = geom.y[i]
                z_aux = geom.z[i] 
                tries = 0
                while success == False:
                    success = True
                    geom.x[i] = x_aux
                    geom.y[i] = y_aux
                    geom.z[i] = z_aux
                    in_radious = LA.norm(np.array([geom.x[i],geom.y[i],geom.z[i]]))
                    # geom.Displace(i,[random.uniform(-magnitude, magnitude),random.uniform(-magnitude, magnitude),random.uniform(-magnitude,magnitude)])
                    # geom.Displace(i,[random.uniform(-magnitude, magnitude),random.uniform(-magnitude, magnitude),random.uniform(-magnitude,magnitude)+0.05])
                    geom.Displace(i,[random.uniform(-magnitude, magnitude)+0.035355,random.uniform(-magnitude, magnitude)+0.035355,random.uniform(-magnitude,magnitude)])
                    out_radious = LA.norm(np.array([geom.x[i],geom.y[i],geom.z[i]]))
                    geom.x[i] = geom.x[i]*in_radious/out_radious
                    geom.y[i] = geom.y[i]*in_radious/out_radious
                    geom.z[i] = geom.z[i]*in_radious/out_radious
                    distances = geom.Distances(i)
                    for j in range(N_at_poly):
                        if distances[j] < min_distance and i != j:
                            success = False
                            tries = tries + 1
                        if tries == 100:
                            geom.x[i] = x_aux
                            geom.y[i] = y_aux
                            geom.z[i] = z_aux
                            success = True

            pos = np.array(geom.PosAsList())
            asym_vec = np.array([0,0,0])
            for i in range(N_at_poly):
                pos[i] = pos[i]/LA.norm(pos[i])
                asym_vec = asym_vec + pos[i]
            asym_vec = asym_vec / (N_at_poly)
            asym = LA.norm(asym_vec)
            geom_aux = copy.deepcopy(geom)
            geom_aux.RemoveAtoms([N_at_poly,N_at_poly+1,N_at_poly+2])
            if asym > band[0] and asym < band[1]:
                spherical = utils.cart2sph_list(geom_aux.PosAsList())
                for i in range(len(spherical)):

                    theta_idx = int(spherical[i][1]*N_theta/np.pi)
                    phi_idx = int((spherical[i][2]+np.pi)*N_phi/(2*np.pi))
                    geom.ShowStruct()
                    dist[theta_idx,phi_idx] = dist[theta_idx,phi_idx] + 1
        print(l,num_stats)

    name = "Struct_dist_spherical_"+file[:-4]+"_110_S"+str(num_stats)+"_mdist"+str(min_distance)+"_bounds"+str(band[0])+"_"+str(band[1])
    # name = "Sym_break_N"+str(geom.Nat)+"_"+str(num_stats)

    with open('out/'+name+'.txt', 'w') as f:
        for i in range(len(dist)):
            for j in range(len(dist[i])):
                f.write(str(i*np.pi/N_theta)+' '+str(j*2*np.pi/N_phi-np.pi)+' '+str(dist[i][j]/dist.sum())+' '+'\n')