import numpy as np
from numpy import random
from geohandler import Handler
from heapq import nsmallest
from operator import itemgetter
import time

class Sphere(Handler):
   """
    Represents a spherical molecular structure with equidistant atoms.

    Inherits from Handler and uses its methods to manipulate the structure.

    Methods:
    SetPos: Overrides the Handler method to set the positions of atoms in a spherical arrangement.
    """
    def __init__(self, Nat,R0):
        super().__init__(Nat,R0)

    def SetPos(self,Nat,R0): #creates a sphere with N equidistant atoms and interatomic distance R0
        self.Nat = Nat
        self.R0 = R0

        indices = np.arange(0, Nat, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/Nat)
        theta = np.pi * (1 + 5**0.5) * indices
        self.x, self.y, self.z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);
        self.R0s, self.widths = self.GetR0s(self.R0neighbours)
        norm=np.mean(self.R0s)/self.R0

        self.x = self.x/norm
        self.y = self.y/norm
        self.z = self.z/norm
        self.x = self.x.tolist()
        self.y = self.y.tolist()
        self.z = self.z.tolist()
        self.R0s = self.R0s/norm
        self.widths = self.widths/norm

class Ring(Handler):
    """
    Represents a ring molecular structure with equidistant atoms.

    Inherits from Handler and uses its methods to manipulate the structure.

    Methods:
    SetPos: Overrides the Handler method to set the positions of atoms in a ring arrangement.
    """

    def __init__(self, Nat,R0):
        super().__init__(Nat,R0)

    def SetPos(self,Nat,R0): #creates a ring with N equidistant atoms and interatomic distance R0
        self.Nat = Nat
        self.R0 = R0

        indices = np.arange(0, Nat, dtype=float)
        for i in range(len(indices)):
            indices[i]=indices[i]+random.rand()/(10*self.R0)

        Dtheta = 2*np.pi /self.Nat

        distance = np.sqrt(2-2*np.cos(Dtheta))
        Radius = self.R0/distance
        self.x, self.y, self.z = Radius*np.cos(indices*Dtheta) ,Radius*np.sin(indices*Dtheta), indices*0;
        self.x = self.x.tolist()
        self.y = self.y.tolist()
        self.z = self.z.tolist()
        self.R0s, self.widths = self.GetR0s(self.R0neighbours)

    def GetR0s(self,Nneighbours): #returns a list of R0 estimations and errors from every atom considering Nneighbours closest neighbours
        print("Calculating R0s..")
        R0= []
        width= []
        for i in range(self.Nat-1):
            R0.append(np.sqrt((self.x[i+1]-self.x[i])**2+(self.y[i+1]-self.y[i])**2+(self.z[i+1]-self.z[i])**2))
        R0.append(np.sqrt((self.x[self.Nat-1]-self.x[0])**2+(self.y[self.Nat-1]-self.y[0])**2+(self.z[self.Nat-1]-self.z[0])**2))
        return R0, width

class Custom(Handler): #creates a custom carbon structure given the atomic positions pos
    """
    Represents a custom molecular structure defined by the user.

    Inherits from Handler and uses its methods to manipulate the structure.

    Methods:
    SetPos: Overrides the Handler method to set the positions of atoms as specified by the user.
    """
    def __init__(self,pos,propcalc=False,types=None):
        self.Nat = len(pos)
        R0=2.4
        self.x = []
        self.y = []
        self.z = []
        for i in range(self.Nat):
            self.x.append(pos[i][0])
            self.y.append(pos[i][1])
            self.z.append(pos[i][2])
        self.pos_as_nparray = np.array(self.PosAsList())
        super().__init__(self.Nat,R0,propcalc)
        if types == None:
            self.types = []
            for i in range(self.Nat):
                self.types.append('C')
        else:
            self.types = types

    def SetPos(self,Nat,R0):
        self.Nat = Nat
        self.R0 = R0

class FromFile(Handler): #loads a structure from file
    """
    Represents a molecular structure loaded from a file.

    Inherits from Handler and uses its methods to manipulate the structure.

    Methods:
    SetPos: Overrides the Handler method to set the positions of atoms based on the file contents.
    """
    def __init__(self,path,propcalc=False):
        self.Nat = 30
        R0=2.4
        super().__init__(self.Nat,R0,propcalc)

        self.LoadGeometry(path)
        if self.types == False:
            self.types = []
            for i in range(self.Nat):
                self.types.append('C')
        self.pos_as_nparray = np.array(self.PosAsList())

    def ReadyPolyethylene(self,idx,Nneighbours=1000):
        a = self.Distances(idx)

        # idx_aux, _ = zip(*nsmallest(Nneighbours, enumerate(a), key=itemgetter(1)))

        sorted_indices = np.argsort(a)
        idx_aux = sorted_indices[:Nneighbours]

        end_time = time.time()
        ret = Custom(self.PosAsListIdx(idx_aux),types=self.TypesAsListIdx(idx_aux))

        return ret

    def SetPos(self,Nat,R0):
        self.Nat = Nat
        self.R0 = R0

class Chain(Handler):
    """
    Represents a linear chain molecular structure with equidistant atoms.

    Inherits from Handler and uses its methods to manipulate the structure.

    Methods:
    SetPos: Overrides the Handler method to set the positions of atoms in a linear chain arrangement.
    """
    def __init__(self, Nat,R0):
        super().__init__(Nat,R0)
        self.types = []
        for i in range(self.Nat):
            self.types.append('C')

    def SetPos(self,Nat,R0): #creates a carbon chain with N equidistant atoms and interatomic distance R0
        self.Nat = Nat
        self.R0 = R0

        indices = np.arange(0, self.Nat, dtype=float)
        dx = []
        dx = R0*(1+0.*((indices-float(self.Nat)/2.0)/(float(self.Nat)/2.0))*((indices-float(self.Nat)/2.0)/(float(self.Nat)/2.0)))
        self.x = []

        for i in range(len(dx)):
            cummulative = 0
            for k in range(i):
                cummulative = cummulative + dx[k]
            self.x.append(cummulative)

        self.y, self.z =0*indices, 0*indices;
        self.y = self.y.tolist()
        self.z = self.z.tolist()
        self.R0s, self.widths = self.GetR0s(self.R0neighbours)

    def GetR0s(self,Nneighbours): #returns a list of R0 estimations and errors from every atom considering Nneighbours closest neighbours
        print("Calculating R0s..")
        R0= []
        width= []
        for i in range(self.Nat-1):
            R0.append(self.x[i+1]-self.x[i])
        R0.append(self.x[self.Nat-1]-self.x[self.Nat-2])
        return R0, width

class ChainCapped(Handler):
    """
    Represents a linear chain molecular structure with equidistant atoms, capped with hydrogen atoms.

    Inherits from Handler and uses its methods to manipulate the structure.

    Methods:
    SetPos: Overrides the Handler method to set the positions of atoms in a linear chain arrangement with hydrogen caps.
    """
    def __init__(self, Nat,R0):
        super().__init__(Nat,R0)
        self.types = []
        self.types.append('H')
        for i in range(self.Nat-2):
            self.types.append('C')
        self.types.append('H')
        # print(self.types)

    def SetPos(self,Nat,R0): #creates a carbon chain with N equidistant atoms and interatomic distance R0, capped with hydrogens
        self.Nat = Nat
        self.R0 = R0

        indices = np.arange(0, self.Nat, dtype=float)
        dx = []
        dx = R0*(1+0.*((indices-float(self.Nat)/2.0)/(float(self.Nat)/2.0))*((indices-float(self.Nat)/2.0)/(float(self.Nat)/2.0)))
        self.x = []

        for i in range(len(dx)):
            cummulative = 0
            for k in range(i):
                cummulative = cummulative + dx[k]
            self.x.append(cummulative)


        self.y, self.z =0*indices, 0*indices;
        self.y = self.y.tolist()
        self.z = self.z.tolist()
        self.R0s, self.widths = self.GetR0s(self.R0neighbours)

    def GetR0s(self,Nneighbours): #returns a list of R0 estimations and errors from every atom considering Nneighbours closest neighbours
        print("Calculating R0s..")
        R0= []
        width= []
        for i in range(self.Nat-1):
            R0.append(self.x[i+1]-self.x[i])
        R0.append(self.x[self.Nat-1]-self.x[self.Nat-2])
        return R0, width