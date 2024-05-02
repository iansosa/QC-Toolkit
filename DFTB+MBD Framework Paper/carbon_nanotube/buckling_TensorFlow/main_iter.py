import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
use_GPU = True
if not use_GPU:
    print('not using GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import time
import numpy as np
import tensorflow as tf
ang = 1/0.529177249
from scipy import optimize
from numpy import linalg as LA
from bond_tf_nanotube import *
from mbd_tf import MBDEvaluator
from pw_tf import PWEvaluator
from NonlinearSolver import *
from scipy.sparse.linalg import cg, gmres

d0 = 0
dt = 0.05
it_max = 1
it_range = range(0, it_max)

tube_choice = 2
vdW_choice = 'pw'  # 'pw' or 'mbd'

# variants of vdW models used in the paper
pw_choice = 'TS'
mbd_choice = 'ts'

optimized = True
use_dihedral = True

# Folder name
if use_dihedral:
    data_folder1 = '/output_data_dihedral'
else:
    data_folder1 = '/output_data'

if optimized:
    data_folder2 = '_opt'
else:
    data_folder2 = ''
data_folder = 'tube%d_' % tube_choice + vdW_choice + data_folder1 + data_folder2

if vdW_choice == 'pw':
    data_folder += '_' + pw_choice
if vdW_choice == 'mbd':
    data_folder += '_' + mbd_choice
print('saving to folder: '+data_folder)

# Basic Geo
Natm_list = [400,640,800]
Nface_list = [10,16,20]
Natm = Natm_list[tube_choice-1]
Nface = Nface_list[tube_choice-1]

# Boundary condition
Ndof = 3*Natm - Nface*6
ListDOF = np.zeros(Ndof, dtype=int)
for i in range(0, Ndof):
    ListDOF[i] = Nface*3 + i

## For the initial optimization ###
#print('initial relaxation...')
#Ndof = 3*Natm - Nface
#ListDOF = np.zeros(Ndof, dtype=int)
#for i in range(0, Nface):
#    ListDOF[2*i] = 3*i
#    ListDOF[2*i+1] = 3*i+1

#for i in range(2*Nface, Ndof):
#    ListDOF[i] = Nface + i

XX0 = np.loadtxt(data_folder + '/XX_d0.000.txt')
YY0 = np.loadtxt(data_folder + '/YY_d0.000.txt')
ZZ0 = np.loadtxt(data_folder + '/ZZ_d0.000.txt')

# structure info. of initial configuration
if vdW_choice == 'wvdW':
    bond_info = bond_search(XX0,YY0,ZZ0,2,output_neighbour_list=False)
    bend_info = bend_search(XX0,YY0,ZZ0,2)
    bond_id = bond_info[:, 0:2].astype(int) - 1
    bend_id = bend_info[:, 0:3].astype(int) - 1
    bond_r0 = bond_info[:, 2]
    bend_theta0 = bend_info[:, 3]
    phi0_list = dihedral_measure(list(zip(XX0, YY0, ZZ0)), bond_id, neighbour_search(bond_id))
    np.savetxt('tube%d_' % tube_choice + vdW_choice + '/bond_info.txt', bond_info)
    np.savetxt('tube%d_' % tube_choice + vdW_choice + '/bend_info.txt', bend_info)
    np.savetxt('tube%d_' % tube_choice + vdW_choice + '/phi0_list.txt', phi0_list)

else:
    bond_info = np.loadtxt('tube%d_' % tube_choice + vdW_choice + '/bond_info.txt')
    bend_info = np.loadtxt('tube%d_' % tube_choice + vdW_choice + '/bend_info.txt')
    phi0_list = np.loadtxt('tube%d_' % tube_choice + vdW_choice + '/phi0_list.txt')
    bond_id = bond_info[:,0:2].astype(int)-1
    bend_id = bend_info[:,0:3].astype(int)-1
    bond_r0 = bond_info[:,2]
    bend_theta0 = bend_info[:,3]

# MBD parameters, scaled by volume
ratio = np.array([0.87 for i in range(0, Natm)])
alpha_0 = [12 for i in range(0, Natm)] * ratio
C6 = [46.6 for i in range(0, Natm)] * ratio ** 2
R_vdw = [3.59 for i in range(0, Natm)] * ratio ** (1/3)

# PW parameters
eps = np.array([1.227e-04 for i in range(0, Natm)])
sigma = np.array([3.4121*1.889725989 for i in range(0, Natm)])

def f(x, ni):
    xbc[ListDOF] = x
    coords = np.array([[xbc[3 * i], xbc[3 * i + 1], xbc[3 * i + 2]] for i in range(0, int(len(xbc) / 3))])

    Ebond, Fbond = Ecov_nanotube_assemble(coords, bond_id, bond_r0*ang, 0.3607, hessian=False)
    Ebend, Fbend = Ebend_nanotube_assemble(coords, bend_id, bend_theta0, 0.2428, hessian=False)
    if use_dihedral:
        Edihedral, Fdihedral = Edihedral_nanotube(coords, bond_id, neighbour_search(bond_id), phi0_list, 0.0194, hessian=False)
    else:
        Fdihedral = Fbond * 0
        Edihedral = 0 
    if vdW_choice == 'wvdW':
        FvdW = Fbond * 0
    elif vdW_choice == 'pw':
        pw = PWEvaluator(hessians=False,model=pw_choice)
        EvdW, FvdW = pw(coords, eps, sigma, alpha_0, C6, R_vdw)
    elif vdW_choice == 'mbd':
        mbd = MBDEvaluator(hessians=False,method=mbd_choice)
        EvdW, FvdW = mbd(coords, alpha_0, C6, R_vdw)

    Etot = Ebond + Ebend + EvdW + Edihedral
    FvdW = FvdW.ravel()
    Fbond = Fbond.ravel()
    Fbend = Fbend.ravel()
    Fdihedral = Fdihedral.ravel()

    Ftot = - Fbond - Fbend - Fdihedral - FvdW

    # Force are saved in terms of Hatree/Bohr
    np.savetxt(data_folder + '/fx_d%.5f.txt' % d, Ftot[0:Natm*3:3])
    np.savetxt(data_folder + '/fy_d%.5f.txt' % d, Ftot[1:Natm*3:3])
    np.savetxt(data_folder + '/fz_d%.5f.txt' % d, Ftot[2:Natm*3:3])
    print(' NORM2 = ', LA.norm(np.array(Fbond[ListDOF]) + Fbend[ListDOF] + Fdihedral[ListDOF]+FvdW[ListDOF]),
          ' Energy = ',np.array(Etot))
    return np.array(Fbond[ListDOF] + Fbend[ListDOF] + Fdihedral[ListDOF] + FvdW[ListDOF])

def gradf(x):
    xbc[ListDOF] = x
    coords = np.array([[xbc[3 * i], xbc[3 * i + 1], xbc[3 * i + 2]] for i in range(0, int(len(xbc) / 3))])

    Kbond = Ecov_nanotube_assemble(coords, bond_id, bond_r0*ang, 0.3607, hessian=True)
    Kbend = Ebend_nanotube_assemble(coords, bend_id, bend_theta0, 0.2428, hessian=True)
    if use_dihedral:
        Kdihedral = Edihedral_nanotube(coords, bond_id, neighbour_search(bond_id), phi0_list, 0.0194, hessian=True)
    else:
        Kdihedral = Kbond * 0

    if vdW_choice == 'wvdW':
        hessian = Kbond * 0
    elif vdW_choice == 'pw':
        pw2 = PWEvaluator(hessians=True,model=pw_choice)
        hessian = pw2(coords, eps, sigma, alpha_0,C6, R_vdw)
    elif vdW_choice == 'mbd':
        mbd2 = MBDEvaluator(hessians=True,method=mbd_choice)
        hessian = mbd2(coords, alpha_0, C6, R_vdw)

    Kbond = np.vstack(np.stack(np.stack(np.vstack(Kbond), axis=2)))
    Kbend = np.vstack(np.stack(np.stack(np.vstack(Kbend), axis=2)))
    Kdihedral = np.vstack(np.stack(np.stack(np.vstack(Kdihedral), axis=2)))
    hessian = np.vstack(np.stack(np.stack(np.vstack(hessian), axis=2)))

    Kbond = np.array([Kbond[ListDOF][i][ListDOF] for i in range(0, Ndof)])
    Kbend = np.array([Kbend[ListDOF][i][ListDOF] for i in range(0, Ndof)])
    Kdihedral = np.array([Kdihedral[ListDOF][i][ListDOF] for i in range(0, Ndof)])
    hessian = np.array([hessian[ListDOF][i][ListDOF] for i in range(0, Ndof)])
    return Kbond + Kbend + Kdihedral + hessian


it = 0
d = d0
while it < it_max:
    print('tube%d_' % tube_choice + vdW_choice)
    try:
        xx = np.loadtxt(data_folder + '/XX_d%.5f.txt' % d)
        yy = np.loadtxt(data_folder + '/YY_d%.5f.txt' % d)
        zz = np.loadtxt(data_folder + '/ZZ_d%.5f.txt' % d)
    except:
        xx = np.loadtxt(data_folder + '/XX_d%.3f.txt' % d)
        yy = np.loadtxt(data_folder + '/YY_d%.3f.txt' % d)
        zz = np.loadtxt(data_folder + '/ZZ_d%.3f.txt' % d)
    xx = np.array(xx[0:Natm*3]*ang)
    yy = np.array(yy[0:Natm*3]*ang)
    zz = np.array(zz[0:Natm*3]*ang)

    try:
        d = d + dt
        print('it = %d, dt = %.5f, d = %.5f' % (it, dt, d))

        for i in range(Nface, Natm):
            zz[i] -= dt * (zz[i] / zz[-1]) * ang

        xbc = list(zip(xx, yy, zz))
        xbc = np.stack(xbc, axis=0).ravel()
        x0 = xbc[ListDOF]
        #######################
        time_start = time.time()
        # root = optimize.fsolve(f, x0, fprime=gradf, xtol=1e-9,args=it)
        root = nonlinear_solve(gradf, f, x0, linear_solver=cg, args=it, maxIters=30, normTolerance=200)
        time_end = time.time()
        print('1 step cost', time_end - time_start)

        #######################
        xbc[ListDOF] = root
        np.savetxt(data_folder + '/XX_d%.5f.txt' % d, xbc[0:Natm * 3:3] / ang)
        np.savetxt(data_folder + '/YY_d%.5f.txt' % d, xbc[1:Natm * 3:3] / ang)
        np.savetxt(data_folder + '/ZZ_d%.5f.txt' % d, xbc[2:Natm * 3:3] / ang)
        # if dt < 0.05:
        #     dt = dt * 2
        it = it + 1
    except:
        d = d - dt
        dt = dt/2
        # if dt > 0.001:
        #     dt = dt/2
