import numpy as np
import sys
import os
import shutil
import structures
from filetypes import SaveGeometry, Loadxyz_raw, Loadgen
from mdhandler import MDHandler
ang = 1/0.529177249

vdW = 'TS'  # 'MBD' or 'TS'
temp = 300
dist = 5
seed = 10
Niter = 5
para = 0  # can set extra subfolders to run difference cases in parallel, i.e. DFTB_para1
folder_path = 'dynamic/' + vdW + '_seed%d/' % seed

# Load geo
_, geo_full, _, atom_types, _ = Loadgen('chains_'+vdW+'.gen',conversion=1)
geo_full = np.array(geo_full).T

geo_single = geo_full[0:30]
geo0 = np.concatenate((geo_single,geo_single+np.array([0,0,dist])))
# print(geo0)
atom_type = [2, ] + [1, ] * 28 + [2, 2] + [1, ] * 28 + [2, ]

# initialize
SaveGeometry(np.array(geo0), atom_type, file_name=folder_path + 'DFTB+'+('_para%d' % para)*(para!=0)+'/geo_0.gen')
chain = structures.Chain(10, 1.42*ang)
chain.LoadGeometry(path='geo_0.gen', dir=folder_path + 'DFTB+'+('_para%d' % para)*(para!=0)+'/')
chain.SaveGeometry(dir=folder_path,para=para)

#
BCs = [0,29,30,59]  # These were manually set into the in_file
chain_MD = MDHandler(chain)
chain_MD.RunMD(Niter,dir=folder_path,temp=temp,vdw=vdW,loadV=False,seed=seed,para=para)

# save
shutil.copy2(folder_path + 'DFTB+' + ('_para%d' % para) * (para != 0)+'/geo_end.xyz', folder_path + 'DFTB+/geo')
os.rename(folder_path+'DFTB+/geo/geo_end.xyz',folder_path + 'DFTB+/geo/geo_end_h%dB_T%d_Niter%d.xyz' % (dist*ang,temp,Niter))
