import numpy as np
import sys
import structures
from filetypes import SaveGeometry
ang = 1/0.529177249

vdW = 'TS'
dist = 5
seed = 10
Niter = 5
para = 0 # can set extra subfolders to run difference cases in parallel

folder_path = 'dynamic/' + vdW + '_seed%d/' % seed

atom_type = [2, ] + [1, ] * 28 + [2, 2] + [1, ] * 28 + [2, ]
file_name = 'dynamic/'+vdW+'_seed%d/DFTB+/geo/geo_end_h%dB_Niter%d.xyz' % (seed,dist*ang,Niter)
print("Loadingxyz..")
try:
    file = open(file_name, "r+")
except OSError:
    print("Could not open xyz file")
    sys.exit()
lines = file.readlines()

aux = lines[0].split(' ')
aux = list(filter(lambda x: x != '', aux))
Nat = int(aux[0])
Niter = int(len(lines) / (2 + Nat))
F_full = np.zeros([Niter*Nat, 3], dtype=np.float32)
for i in range(0,Niter):
    print('processing i =', i)
    instant = []
    for j in range(Nat):
        a = lines[2 + j + i * (2 + Nat)].split(' ')
        a = list(filter(lambda x: x != '', a))
        a = list(map(float,a[1:4]))
        instant.append(a)
    instant = np.array(instant)
    SaveGeometry(instant, atom_type, file_name=folder_path + 'DFTB+'+('_para%d' % para)*(para!=0)+'/geo_temp.gen')
    chain = structures.Chain(10, 1.42*ang)
    chain.LoadGeometry(path='geo_temp.gen', dir=folder_path + 'DFTB+'+('_para%d' % para)*(para!=0)+'/')
    chain.SaveGeometry(dir=folder_path,para=para)
    chain.RunStatic(vdW,dir=folder_path,para=para)

    F = chain.GetForces(dir=folder_path,para=para)
    F_full[i*Nat:(i+1)*Nat,:] = F

np.savetxt('data/F_'+vdW+'_h%dB_seed%d_Niter%d.txt' % (dist*ang,seed,Niter-1), F_full)