import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
use_GPU = True
if not use_GPU:
    print('not using GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
ang = 1/0.529177249
from mbd_tf import MBDEvaluator
from pw_tf import PWEvaluator

C1_list = np.linspace(2,100,50,dtype=int)
NatmC2 = int(200)
h = 20

###
pw_TS = PWEvaluator(hessians=False, model='TS')
mbd_ts = MBDEvaluator(hessians=False, method='ts')

for NatmC1 in C1_list:
    print('NatmC1 =',NatmC1)
    Natm = NatmC1 + NatmC2

    # Initialize the C chain
    XC1 = np.linspace(0, (NatmC1 - 1) * 1.2, NatmC1)
    YC1 = h * np.ones(NatmC1)
    ZC1 = np.zeros(NatmC1)
    XC2 = np.linspace((NatmC2 - 1) * 1.2, 0, NatmC2)
    YC2 = np.zeros(NatmC2)
    ZC2 = np.zeros(NatmC2)

    XC1 = XC1 - np.average(XC1)
    XC2 = XC2 - np.average(XC2)
    # Combine C and Si
    XX0 = np.concatenate((XC1, XC2)) * ang
    YY0 = np.concatenate((YC1, YC2)) * ang
    ZZ0 = np.concatenate((ZC1, ZC2)) * ang
    xbc = list(zip(XX0, YY0, ZZ0))
    xbc = np.stack(xbc, axis=0).ravel()

    # MBD parameters, from DFTB
    if NatmC1 == 1:
        ratio_list_1 = np.array([1])
    else:
        ratio_list_1 = np.array([0.78, ] * NatmC1)
    ratio_list_2 = np.array([0.78, ] * NatmC2)
    ratio_list = np.concatenate((ratio_list_1, ratio_list_2))
    alpha_0 = [12 for i in range(0, Natm)] * ratio_list
    C6 = [46.6 for i in range(0, Natm)] * ratio_list ** 2
    R_vdw = [3.59 for i in range(0, Natm)] * ratio_list ** (1 / 3)

    # PW parameters
    eps = [1.227e-04 for i in range(0, Natm)]
    sigma = [3.4121 * 1.889725989 for i in range(0, Natm)]

    coords = np.array([[xbc[3 * i], xbc[3 * i + 1], xbc[3 * i + 2]] for i in range(0, int(len(xbc) / 3))])

    EvdW_TS, FvdW_TS = pw_TS(coords, eps, sigma, alpha_0,C6, R_vdw)
    EvdW_ts, FvdW_ts = mbd_ts(coords, alpha_0, C6, R_vdw)

    np.savetxt('data/Fvdw_NC%d_NCsub%d_h%.2f_PW_TS.txt' % (NatmC1, NatmC2, h), -FvdW_TS)
    np.savetxt('data/Fvdw_NC%d_NCsub%d_h%.2f_MBD_ts.txt' % (NatmC1, NatmC2, h), -FvdW_ts)
