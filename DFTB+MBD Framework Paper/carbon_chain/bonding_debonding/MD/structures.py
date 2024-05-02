import numpy as np
from numpy import random
from geohandler import Handler
from mdhandler import MDHandler
class Sphere(Handler):

    def __init__(self, Nat,R0):
        super().__init__(Nat,R0)

    def SetPos(self,Nat,R0): #creates a sphere with N equidistant atoms and interatomic distance R0
        self.Nat = Nat
        self.R0 = R0

        indices = np.arange(0, Nat, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/Nat)
        theta = np.pi * (1 + 5**0.5) * indices
        self.x, self.y, self.z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
        self.R0s, self.widths = self.GetR0s(self.R0neighbours)
        norm= np.mean(self.R0s)/self.R0

        self.x = self.x/norm
        self.y = self.y/norm
        self.z = self.z/norm
        self.x = self.x.tolist()
        self.y = self.y.tolist()
        self.z = self.z.tolist()
        self.R0s = self.R0s/norm
        self.widths = self.widths/norm


class Ring(Handler):

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
        self.x, self.y, self.z = Radius*np.cos(indices*Dtheta) ,Radius*np.sin(indices*Dtheta), indices*0
        self.x = self.x.tolist()
        self.y = self.y.tolist()
        self.z = self.z.tolist()
        self.R0s, self.widths = self.GetR0s(self.R0neighbours)

    def GetR0s(self,Nneighbours):  # returns a list of R0 estimations and errors from every atom considering Nneighbours closest neighbours
        print("Calculating R0s..")
        R0 = []
        width= []
        for i in range(self.Nat-1):
            R0.append(np.sqrt((self.x[i+1]-self.x[i])**2+(self.y[i+1]-self.y[i])**2+(self.z[i+1]-self.z[i])**2))
        R0.append(np.sqrt((self.x[self.Nat-1]-self.x[0])**2+(self.y[self.Nat-1]-self.y[0])**2+(self.z[self.Nat-1]-self.z[0])**2))
        return R0, width

class Custom(Handler):

    def __init__(self,pos,propcalc=False):
        self.Nat = len(pos)
        R0=2.4
        self.x = []
        self.y = []
        self.z = []
        for i in range(self.Nat):
            self.x.append(pos[i][0])
            self.y.append(pos[i][1])
            self.z.append(pos[i][2])

        super().__init__(self.Nat,R0,propcalc)

    def SetPos(self,Nat,R0):
        self.Nat = Nat
        self.R0 = R0


class Chain(Handler):

    def __init__(self, Nat,R0):
        super().__init__(Nat,R0)

    def SetPos(self,Nat,R0): #creates a chain with N equidistant atoms and interatomic distance R0
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

        # dx = []
        # dz = []
        # self.x = []
        # self.x.append(0)
        # self.x.append(R0)
        # self.z = []
        # self.z.append(0)
        # self.z.append(0)

        # for i in range(2,self.Nat):
        #     cummulativex = R0-0.92387*R0
        #     cummulativez = -0.38268*R0
        #     for k in range(i):
        #         cummulativex = cummulativex + 0.92387*R0
        #         cummulativez = cummulativez + 0.38268*R0
        #     self.x.append(cummulativex)
        #     self.z.append(cummulativez)


        self.y, self.z =0*indices, 0*indices
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

