import sys
import numpy as np
from numpy.linalg import norm as norm
from scipy.special import erf as erf
from pymbd import mbd_energy as MBDcalc_Py, from_volumes
from tqdm import tqdm
import random

default = {
            'code':'pymbd-f',        # implementation
            'beta':0.83,             # MBD damping factor
            'Sr':0.94,               # TS  damping factor
            'screening':'rsscs',     # MBD screening mode
            # 'screening':'plain',     # MBD screening mode
            'parameters':'default',  # vdW parameters
          }

class vdWclass:

    valid_args = ['code',\
                  'beta',\
                  'Sr',\
                  'screening',\
                  'parameters',\
                 ]

    valid_models = ['MBD','TS']

    def _set_calc(self):
        if self.code == 'pymbd-f':
            from pymbd.fortran import MBDGeom as evaluator
        elif self.code == 'tf':
            print('WARNING: tf implementation not tested')
            from calcs.mbd_tf import MBDEvaluator as evaluator
        else:
            raise RuntimeError('uknown code ', self.code)

        return evaluator

    def __init__(self, **kwargs):

        for arg, val in default.items():
                setattr(self, arg, val)

        self.calc = self._set_calc()


    def _combineC6(self):
       
        r = self.a0[:,None]/self.a0[None,:]
        d = self.C6[:,None]*r + self.C6[None,:]/r
        return 2*self.C6[:,None]*self.C6[None,:]/d

    def _get_energy(self, calc):
        out = calc.mbd_energy(self.a0, self.C6, self.Rvdw, self.beta,force = False, variant = self.screening)
        return out

    def _get_forces(self, calc,discrete=False):

        if self.model == 'MBD':
            E, F = calc.mbd_energy(self.a0, self.C6, self.Rvdw, self.beta,force = True, variant = self.screening)
            return -F
        elif self.model == 'TS':
            d = 20.
            C6ij = self._combineC6()
            beta = self.Sr*(self.Rvdw[:,None]+self.Rvdw[None,:])
            # print(self.Rvdw[:,None])
            # print(beta)
            pos = np.array(self.pos)
            Rij = []
            rij = []
            for i in range(len(self.pos)):
                Rij_row = []
                rij_row = []
                for j in range(len(self.pos)):
                    Rij_row.append(pos[i]-pos[j])
                    rij_row.append(np.linalg.norm(pos[i]-pos[j]))
                Rij.append(np.array(Rij_row))
                rij.append(np.array(rij_row))
            Rij = np.array(Rij)
            rij = np.array(rij)
            f = 1./(1. + np.exp(-d*(rij/beta-1.)))
            temp = np.copy(rij)
            temp[temp==0] = 1e20
            invrij = temp**(-1)
            Eij = - f*C6ij*invrij**(6)
            bracket = d/beta*f*np.exp(-d*(rij/beta-1.)) - 6.*invrij
            Fij = bracket[:,:,None]*Eij[:,:,None]*invrij[:,:,None]*Rij
            if discrete == False:
                return -np.sum(Fij,axis=1)
            else:
                return -Fij
        

    def _get_hessian(self,db=0.01):
        Nat = len(self.pos)
        calc = self.calc(self.pos)
        eqForces = self._get_forces(calc)

        Hessian = np.zeros((Nat*3,Nat*3))
        for_desc = "Hessian calculation for " + self.model + "..."
        for i in tqdm (range(Nat), desc=for_desc):

            ############x##############
            self.pos[i][0] = self.pos[i][0] + db
            calc = self.calc(self.pos)
            F = self._get_forces(calc)
            for j in range(Nat):
                Hessian[3*i][3*j]=Hessian[3*i][3*j]+(F[j][0]-eqForces[j][0])/(2*db)
                Hessian[3*i][3*j+1]=Hessian[3*i][3*j+1]+(F[j][1]-eqForces[j][1])/(2*db)
                Hessian[3*i][3*j+2]=Hessian[3*i][3*j+2]+(F[j][2]-eqForces[j][2])/(2*db)
            self.pos[i][0] = self.pos[i][0] - 2*db
            calc = self.calc(self.pos)
            F = self._get_forces(calc)
            for j in range(Nat):
                Hessian[3*i][3*j]=Hessian[3*i][3*j]-(F[j][0]-eqForces[j][0])/(2*db)
                Hessian[3*i][3*j+1]=Hessian[3*i][3*j+1]-(F[j][1]-eqForces[j][1])/(2*db)
                Hessian[3*i][3*j+2]=Hessian[3*i][3*j+2]-(F[j][2]-eqForces[j][2])/(2*db)
            self.pos[i][0] = self.pos[i][0] + db
            ############y############
            self.pos[i][1] = self.pos[i][1] + db
            calc = self.calc(self.pos)
            F = self._get_forces(calc)
            for j in range(Nat):
                Hessian[3*i+1][3*j]=Hessian[3*i+1][3*j]+(F[j][0]-eqForces[j][0])/(2*db)
                Hessian[3*i+1][3*j+1]=Hessian[3*i+1][3*j+1]+(F[j][1]-eqForces[j][1])/(2*db)
                Hessian[3*i+1][3*j+2]=Hessian[3*i+1][3*j+2]+(F[j][2]-eqForces[j][2])/(2*db)
            self.pos[i][1] = self.pos[i][1] - 2*db
            calc = self.calc(self.pos)
            F = self._get_forces(calc)
            for j in range(Nat):
                Hessian[3*i+1][3*j]=Hessian[3*i+1][3*j]-(F[j][0]-eqForces[j][0])/(2*db)
                Hessian[3*i+1][3*j+1]=Hessian[3*i+1][3*j+1]-(F[j][1]-eqForces[j][1])/(2*db)
                Hessian[3*i+1][3*j+2]=Hessian[3*i+1][3*j+2]-(F[j][2]-eqForces[j][2])/(2*db)
            self.pos[i][1] = self.pos[i][1] + db
            ############z############
            self.pos[i][2] = self.pos[i][2] + db
            calc = self.calc(self.pos)
            F = self._get_forces(calc)
            for j in range(Nat):
                Hessian[3*i+2][3*j]=Hessian[3*i+2][3*j]+(F[j][0]-eqForces[j][0])/(2*db)
                Hessian[3*i+2][3*j+1]=Hessian[3*i+2][3*j+1]+(F[j][1]-eqForces[j][1])/(2*db)
                Hessian[3*i+2][3*j+2]=Hessian[3*i+2][3*j+2]+(F[j][2]-eqForces[j][2])/(2*db)
            self.pos[i][2] = self.pos[i][2] - 2*db
            calc = self.calc(self.pos)
            F = self._get_forces(calc)
            for j in range(Nat):
                Hessian[3*i+2][3*j]=Hessian[3*i+2][3*j]-(F[j][0]-eqForces[j][0])/(2*db)
                Hessian[3*i+2][3*j+1]=Hessian[3*i+2][3*j+1]-(F[j][1]-eqForces[j][1])/(2*db)
                Hessian[3*i+2][3*j+2]=Hessian[3*i+2][3*j+2]-(F[j][2]-eqForces[j][2])/(2*db)
            self.pos[i][2] = self.pos[i][2] + db
        return Hessian

    def calculate(self, pos, what='energy', model="MBD",discrete=False,atom_types = None, volumes = None):

        self.pos = pos
        if atom_types == None:
            atom_types = []
            for i in range(len(pos)):
                atom_types.append("C")
        if volumes == None:
            volumes = []
            for i in range(len(atom_types)):
                if atom_types[i] == "C":
                    volumes.append(0.7904) #polymer melts
                    # volumes.append(0.87)
                if atom_types[i] == "H":
                    volumes.append(0.6167)
            # self.a0, self.C6, self.Rvdw = from_volumes(atom_types, 0.87)

        self.a0, self.C6, self.Rvdw = from_volumes(atom_types,volumes)
        if model not in self.valid_models:
            raise ValueError('model'+model + 'not known')
        self.model = model

        calc = self.calc(self.pos)

        outs = []
        for task in what:
            if   task == 'energy':
                outs.append(self._get_energy(calc))
            elif task == 'forces':
                outs.append(self._get_forces(calc,discrete))
            elif task == 'hessian':
                outs.append(self._get_hessian())
            else:
                raise RuntimeError(task, ' is not implemented')
        if len(what) == 1: outs=outs[0]

        return outs

