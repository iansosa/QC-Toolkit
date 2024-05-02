import structures
from filetypes import Loadgen, SaveGeometry
from geohandler import Handler as GH
import numpy as np
import numpy.linalg as LA
import time
from utils import GetStress
ang = 1/0.529177249

h0 = 5
vdW_choice = 'TS'
NC1 = 28
NC2 = 28
NatmC = NC1 + NC2
du = 0.2
d_range = range(0,10)
load = 1

folder_path = 'relaxation/' + vdW_choice + '/'
# geo0_name = 'NC%d_NC%d_h%.2f_load%d.gen' % (NC1, NC2, h0, load) # for resume
geo0_name = 'NC%d_NC%d_h%.2f.gen' % (NC1,NC2,h0)

# Load the initial geo
chain = structures.Chain(NatmC, 1.42*ang)
chain.LoadGeometry(path=geo0_name, dir=folder_path+'DFTB+/geo/')

# constraints upper chain BCs
constraints = [[NatmC,'x'],[NatmC,'y'], [NatmC,'z'],
               [NatmC+1,'x'],[NatmC+1,'y'],[NatmC+1,'z'],
               [NatmC+2,'x'],[NatmC+2,'y'],[NatmC+2,'z'],
               [NatmC+3,'x'],[NatmC+3,'y'],[NatmC+3,'z']]
# Relax
for i in d_range:
    h = h0 + load*i*du
    geo_name = 'NC%d_NC%d_h%.2f_load%d' % (NC1, NC2, h, load)

    chain.Displace(NatmC, du * ang * np.array([0, load, 0]) * (i != 0))
    chain.Displace(NatmC+1, du * ang * np.array([0, load, 0]) * (i != 0))

    chain.SaveGeometry(dir=folder_path)
    chain.RunOptimize(vdw=vdW_choice, constraints=constraints,dir=folder_path)
    chain.LoadGeometry("geo_end.gen",dir=folder_path)
    chain.SaveGeometry(decour=geo_name, path='geo', dir=folder_path)

    F = chain.GetForces(dir=folder_path)
    np.savetxt(folder_path + 'output/F_' + geo_name + '.txt',F)

