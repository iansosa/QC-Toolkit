import numpy as np
import matplotlib.pyplot as plot
import math
import sys
import subprocess
import shutil
import os.path
import os
import copy
import filetypes
import numpy.linalg as LA
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from itertools import permutations 
from scipy.spatial.transform import Rotation as R

class Handler():

    def __init__(self,Nat,R0,propcalc=True): #initialize the desired geometry with interatomic distance R0 (Bohr)
        self.Nat = Nat
        self.R0s = None
        self.R0 = R0 #interatomic distance in bohr
        self.widths = None
        self.R0neighbours = 2 #number of closest neighbours to calculate the R0 approximation
        if self.Nat <= self.R0neighbours:
            self.R0neighbours=self.Nat-1
        self.SetPos(self.Nat,self.R0)
        self.Cutoffneighbours = self.R0*1.1 #cutoffdistance for bonded neighbours
        self.bonds = None
        self.offplane = None
        self.dihedral = None
        self.angles = None #every angle formed by 3 atoms. angles[i] holds all of the angles related to atom i, None if there are none.  angles[i][k][0],angles[i][k][1] indices of two atoms forming angle k. angles[i][k][2] the angle in radians. angles[i][k][3] a perpendicular vector to angles[i][k][0],angles[i][k][1]
        self.distances = None
        if propcalc == True:
            self.CalcBondedNeighbours(self.Cutoffneighbours)
            self.CalcBondDistances()
            self.CalcBondAngles()
            self.CalcBondOffPlane()
            self.CalcBondDihedral()
        self.periodic = False
        self.types = []
        self.unit_cell = []

    def add(self,structure):
        self.Nat = self.Nat+structure.Nat
        # self.x = self.x.tolist()
        # self.y = self.y.tolist()
        # self.z = self.z.tolist()
        for i in range(structure.Nat):
            self.x.append(structure.x[i])
            self.y.append(structure.y[i])
            self.z.append(structure.z[i])
        # self.CalcBondedNeighbours(self.Cutoffneighbours)
        # self.CalcBondDistances()
        # self.CalcBondAngles()
        # self.CalcBondOffPlane()
        # self.CalcBondDihedral()

    def SetPos(self,Nat,R0): #set the position of the geometry
        print ("SetPos Unimplemented")
        sys.exit()

    def Pos(self): #returns the positions of every atom
        return self.x, self.y, self.z
    def PosAsList(self): #returns the positions of every atom as a list
        pos = []
        for i in range(self.Nat):
            pos.append([self.x[i],self.y[i],self.z[i]])
        return pos

    def ShowStruct(self): #displays the 3D structure

        fig = plot.figure(figsize=(5,5))
        ax = fig.gca(projection='3d')

        ###scaling
        x_scale=1
        y_scale=1
        z_scale=1.3
        
        scale=np.diag([x_scale, y_scale, z_scale, 1.0])
        scale=scale*(1.0/scale.max())
        scale[3,3]=1.0

        def short_proj():
          return np.dot(Axes3D.get_proj(ax), scale)

        ax.get_proj=short_proj
        ###end scaling

        ax.scatter(self.x,self.y,self.z)
        plot.show()

    def ShowR0s(self): #displays a list of the R0 estimations for every atom
        x = np.linspace(1,self.Nat,self.Nat)
        plot.scatter(x, self.R0s,color="black")
        plot.show()

    def ShowWidths(self): #displays a list of the errors in the R0 estimations for every atom
        x = np.linspace(1,self.Nat,self.Nat)
        plot.scatter(x, self.widths,color="black")
        plot.show()

    def ShowDistances(self,idx): #displays a list of the errors in the R0 estimations for every atom
        dist=self.Distances(idx)
        dist.sort()
        print(dist)

        x = np.linspace(1,self.Nat,self.Nat)
        plot.scatter(x, dist,color="black")
        plot.show()

    def SaveR0s(self): #Saves R0s to file
        print("Saving R0s..")
        with open('out/R0s.txt', 'w') as f:
            for i in range(len(self.R0s)):
                f.write(str(i)+' '+str(self.R0s[i])+'\n')

    def Distances(self,idx): #returns a list of all the interatomic distances from atom idx
        dist = []
        for i in range(self.Nat):
            dist.append(np.sqrt((self.x[idx]-self.x[i])**2+(self.y[idx]-self.y[i])**2+(self.z[idx]-self.z[i])**2))

        return dist

    def Distance(self,i,j): #returns the distance between atom i and j
        return np.sqrt((self.x[i]-self.x[j])**2+(self.y[i]-self.y[j])**2+(self.z[i]-self.z[j])**2)

    def SaveDistances(self): #Saves all the distances (Bohr) for each atom to a file
        dist = []
        for i in range(self.Nat):
            distances = self.Distances(i)
            distances.sort()
            dist.append(distances)

        print("Saving distances..")
        with open('out/Distances.txt', 'w') as f:
            for i in range(self.Nat):
                f.write(str(i)+' ')
                for k in range(self.Nat):
                    f.write(str(dist[k][i])+' ')
                f.write('\n')

    def GetR0s(self,Nneighbours): #returns a list of R0 estimations and errors from every atom considering Nneighbours closest neighbours
        print("Calculating R0s..")
        R0= []
        width= []
        for i in range(self.Nat):
            dist=self.Distances(i)
            dist.sort()
            median = 0
            for k in range(Nneighbours):
                median= median + dist[k+1]
            median=median/Nneighbours
            R0.append(median)
            width.append(dist[Nneighbours]-dist[1])
        return R0, width

    def UpdateR0s(self): #returns a list of R0 estimations and errors from every atom considering Nneighbours closest neighbours
        self.R0s, self.widths = self.GetR0s(self.R0neighbours)
        self.R0 = np.mean(self.R0s)
        self.CalcBondedNeighbours(self.R0*1.1)
        self.CalcBondDistances()
        self.CalcBondAngles()
        self.CalcBondOffPlane()
        self.CalcBondDihedral()

    def SaveGeometry(self,decour="geom",path=None,instruct=False,charges=None,decour_charges="charges",dir=''): #saves the geometry to a gen file in angstroms
        print("Saving geometry..")
        angstrom = 0.529177249
        folder = dir + "DFTB+"
        if instruct == True:
            folder = "SavedStructures"
        if self.periodic == False:
            name=folder+'/'+decour+'.gen'
            if path != None:
                name = folder+'/'+path+'/'+decour+'.gen'
                if not os.path.exists(folder+'/'+path):
                    os.makedirs(folder+'/'+path)

            with open(name, 'w') as f:
                f.write(str(self.Nat)+' C\n')
                f.write('  C H\n')
                for i in range(self.Nat):
                    if self.types[i] == 'C':
                        atom_type = 1
                    elif self.types[i] == 'H':
                        atom_type = 2
                    f.write('  '+str(i+1)+' ' + str(atom_type)+'  '+str(angstrom*self.x[i])+' '+str(angstrom*self.y[i])+' '+str(angstrom*self.z[i])+'\n')
        elif self.periodic == True:
            name= folder+'/'+decour+'.gen'
            if path != None:
                name = folder+'/'+path+'/'+decour+'.gen'
                if not os.path.exists(folder+'/'+path):
                    os.makedirs(folder+'/'+path)
            with open(name, 'w') as f:
                f.write(str(self.Nat)+' S\n')
                f.write('  C\n')
                for i in range(self.Nat):
                    f.write('  '+str(i+1)+' 1  '+str(angstrom*self.x[i])+' '+str(angstrom*self.y[i])+' '+str(angstrom*self.z[i])+'\n')
                for i in range(len(self.unit_cell)):
                    f.write('  '+str(angstrom*self.unit_cell[i][0])+' '+str(angstrom*self.unit_cell[i][1])+' '+str(angstrom*self.unit_cell[i][2])+'\n')
        if charges != None:
            totalcharge = 0
            for i in range(self.Nat):
                totalcharge=totalcharge+charges[i][6]
            with open("DFTB+/charges.dat", 'w') as f:
                f.write("           6\n")
                f.write(" F F F T          "+str(self.Nat)+"           1   "+str(totalcharge)+"\n")
                for i in range(self.Nat):
                    f.write("   "+str(charges[i][6])+"        0.0000000000000000        0.0000000000000000        0.0000000000000000     \n")
                f.write(" 0 0\n")
            if path != None:
                with open(folder+'/'+path+'/'+decour_charges+".dat", 'w') as f:
                    f.write("           6\n")
                    f.write(" F F F T          "+str(self.Nat)+"           1   "+str(totalcharge)+"\n")
                    for i in range(self.Nat):
                        f.write("   "+str(charges[i][6])+"        0.0000000000000000        0.0000000000000000        0.0000000000000000     \n")
                    f.write(" 0 0\n")



    def LoadGeometry(self,path="geom.out.gen",dir=''): #loads the geometry from a gen, xyz or sdf file in angstroms and converts it into Bohr
        print("Loading geometry..")
        angstrom = 0.529177249
        extension = path[-3:]
        recognized = False
        self.periodic = False

        if extension == "sdf":
            recognized = True
            if path=="Graphene-C92.sdf":
                angstrom = angstrom * 0.71121438
            self.Nat, geometry = filetypes.Loadsdf("SavedStructures/"+path,angstrom)
            print(str(self.Nat)+" atoms loaded")
                
        if extension == "gen":
            recognized = True
            if path != "geom.out.gen" and path != "geo_end.gen":
                self.Nat, geometry, self.periodic, self.types, self.unit_cell = filetypes.Loadgen(dir + path,angstrom)
                print(str(self.Nat)+" atoms loaded")
            else:
                self.Nat, geometry, self.periodic, self.types, self.unit_cell = filetypes.Loadgen(dir + "DFTB+/"+path, angstrom)

        if extension == "xyz":
            recognized = True
            self.Nat, geometry = filetypes.Loadxyz_single("SavedStructures/"+path,angstrom)

        if extension == "cc1":
            recognized = True
            self.Nat, geometry = filetypes.Loadcc1("SavedStructures/"+path,angstrom)

        if extension == "txt":
            recognized = True
            self.Nat, geometry = filetypes.Loadtxt("SavedStructures/"+path,angstrom)
            print(str(self.Nat)+" atoms loaded")

        if recognized == False:
            print ("Extension not recognized")
            sys.exit()

        if self.Nat <= self.R0neighbours:
            self.R0neighbours=self.Nat-1

        if self.periodic == True:
            print("geometry is periodic")
        self.x = geometry[0]
        self.y = geometry[1]
        self.z = geometry[2]
        # self.R0s, self.widths = self.GetR0s(self.R0neighbours)
        # self.R0 = np.mean(self.R0s)
        # self.CalcBondedNeighbours(self.R0*1.1)
        # self.CalcBondDistances()
        # self.CalcBondAngles()
        # self.CalcBondOffPlane()
        # self.CalcBondDihedral()

    def RunOptimize(self,vdw=None,static=None,constraints=None,read_charges=False,packing=[5,5,5],mbdpacking=[3,3,3], fixlengths=False, dir=''):
        if self.periodic == False:
            if vdw == None:
                shutil.copyfile('in_files/optimize.hsd', dir + 'DFTB+/dftb_in.hsd')
            elif vdw == "MBD":
                shutil.copyfile('in_files/optimize_mbd.hsd',dir + 'DFTB+/dftb_in.hsd')
            elif vdw == "PW":
                shutil.copyfile('in_files/optimize_pw.hsd',dir + 'DFTB+/dftb_in.hsd')
            elif vdw == "TS":
                shutil.copyfile('in_files/optimize_ts.hsd',dir + 'DFTB+/dftb_in.hsd')
            else:
                print ("Dispersion type not recognized")
                sys.exit()
            try:
                file = open(dir + "DFTB+/dftb_in.hsd", "r+")
            except OSError:
                print ("Could not open dftb_in.hsd file")
                sys.exit()
        else:
            if vdw == None:
                shutil.copyfile('in_files/optimize-periodic' + '_LatOpt'*(fixlengths!=False) + '.hsd', dir + 'DFTB+/dftb_in.hsd')
            elif vdw == "MBD":
                shutil.copyfile('in_files/optimize-periodic_mbd' + '_LatOpt'*(fixlengths!=False) + '.hsd', dir + 'DFTB+/dftb_in.hsd')
            elif vdw == "PW":
                shutil.copyfile('in_files/optimize-periodic_pw' + '_LatOpt'*(fixlengths!=False) + '.hsd', dir + 'DFTB+/dftb_in.hsd')
            elif vdw == "TS":
                shutil.copyfile('in_files/optimize-periodic_ts' + '_LatOpt'*(fixlengths!=False) + '.hsd', dir + 'DFTB+/dftb_in.hsd')
            else:
                print ("Dispersion type not recognized")
                sys.exit()
            try:
                file = open(dir + "DFTB+/dftb_in.hsd", "r+")
            except OSError:
                print ("Could not open dftb_in.hsd file")
                sys.exit()
        lines = file.readlines()
        file.close()

        if read_charges == True:
            idx = -1
            for i in range(len(lines)):
                if lines[i].find("ReadInitialCharges") != -1:
                    idx = i
            targetline = "  ReadInitialCharges = Yes\n"
            lines[idx] = targetline

        if self.periodic == True:
            idx = -1
            for i in range(len(lines)):
                if lines[i].find("KPointsAndWeights") != -1:
                    idx = i
            lines[idx+1] = "    "+str(packing[0])+ " 0 0\n"
            lines[idx+2] = "    0 "+str(packing[1])+ " 0\n"
            lines[idx+3] = "    0 0 "+str(packing[2])+"\n"
            shift = "    "
            if packing[0] % 2 == 0:
                shift = shift + "0.5 "
            else:
                shift = shift + "0 "
            if packing[1] % 2 == 0:
                shift = shift + "0.5 "
            else:
                shift = shift + "0 "
            if packing[2] % 2 == 0:
                shift = shift + "0.5\n"
            else:
                shift = shift + "0\n"
            lines[idx+4] = shift

        if self.periodic == True and vdw == "MBD":
            idx = -1
            for i in range(len(lines)):
                if lines[i].find("KGRID") != -1:
                    idx = i
            lines[idx] = "    KGRID = "+str(mbdpacking[0])+" "+str(mbdpacking[1])+" "+str(mbdpacking[2])+"\n"

        if self.periodic == True and fixlengths != False:
            idx = -1
            for i in range(len(lines)):
                if lines[i].find("FixLengths") != -1:
                    idx = i
            lines[idx] = "  FixLengths = {" + fixlengths + "}" + "\n"

        if static is not None:
            idx = -1
            for i in range(len(lines)):
                if lines[i].find("MovedAtoms") != -1:
                    idx = i
            targetline = "  MovedAtoms = !("
            for j in range(len(static)):
                targetline = targetline + str(static[j]+1) + " "
            targetline = targetline + ")"+"\n"
            lines[idx] = targetline

        if constraints is not None:
            idx = -1
            for i in range(len(lines)):
                if lines[i].find("MaxSteps") != -1:
                    idx = i+1
            targetline = "  Constraints = {"
            for j in range(len(constraints)):
                if constraints[j][1] == 'x':
                   fix_axis = '1.0 0.0 0.0'
                elif constraints[j][1] == 'y':
                   fix_axis = '0.0 1.0 0.0'
                elif constraints[j][1] == 'z':
                   fix_axis = '0.0 0.0 1.0'

                targetline = targetline + '\n' + '' + str(constraints[j][0]+1) + " " + fix_axis
            targetline = targetline + "}"+"\n}"
            lines[idx] = targetline


        with open(dir + 'DFTB+/dftb_in.hsd', 'w') as f:
            for i in range(len(lines)):
                f.write(lines[i])

        # with open(dir + 'DFTB+/dftbOpt.sh', 'w') as f:
        #     lines = f.readlines()
        #     for i in range(len(lines)):
        #         if lines[i].find("cd") != -1:
        #             lines[i] = 'cd ' + dir + "DFTB+"
        #     for i in range(len(lines)):
        #         f.write(lines[i])

        subprocess.run("./" + dir + "DFTB+/dftbOpt.sh", shell=True)

    def RunStatic(self,vdw=None,read_charges=False,packing=[5,5,5],mbdpacking=[2,2,2],dir=''):
        if self.periodic == False:
            if vdw == None:
                shutil.copyfile('in_files/static_calc.hsd', dir + 'DFTB+/dftb_in.hsd')
            elif vdw == "MBD":
                shutil.copyfile('in_files/static_calc_mbd.hsd', dir + 'DFTB+/dftb_in.hsd')
            elif vdw == "PW":
                shutil.copyfile('in_files/static_calc_pw.hsd', dir + 'DFTB+/dftb_in.hsd')
            elif vdw == "TS":
                shutil.copyfile('in_files/static_calc_ts.hsd', dir + 'DFTB+/dftb_in.hsd')
            else:
                print ("Dispersion type not recognized")
                sys.exit()
        else:
            if vdw == None:
                shutil.copyfile('in_files/static_calc-periodic.hsd', dir + 'DFTB+/dftb_in.hsd')
            elif vdw == "MBD":
                shutil.copyfile('in_files/static_calc-periodic_mbd.hsd', dir + 'DFTB+/dftb_in.hsd')
            elif vdw == "PW":
                shutil.copyfile('in_files/static_calc-periodic_pw.hsd', dir + 'DFTB+/dftb_in.hsd')
            elif vdw == "TS":
                shutil.copyfile('in_files/static_calc-periodic_ts.hsd', dir + 'DFTB+/dftb_in.hsd')
            else:
                print ("Dispersion type not recognized")
                sys.exit()

        try:
            file = open(dir + "DFTB+/dftb_in.hsd", "r+")
        except OSError:
            print ("Could not open dftb_in.hsd file")
            sys.exit()
        lines = file.readlines()
        file.close()

        if self.periodic == True:
            idx = -1
            for i in range(len(lines)):
                if lines[i].find("KPointsAndWeights") != -1:
                    idx = i
            lines[idx+1] = "    "+str(packing[0])+ " 0 0\n"
            lines[idx+2] = "    0 "+str(packing[1])+ " 0\n"
            lines[idx+3] = "    0 0 "+str(packing[2])+"\n"
            shift = "    "
            if packing[0] % 2 == 0:
                shift = shift + "0.5 "
            else:
                shift = shift + "0 "
            if packing[1] % 2 == 0:
                shift = shift + "0.5 "
            else:
                shift = shift + "0 "
            if packing[2] % 2 == 0:
                shift = shift + "0.5\n"
            else:
                shift = shift + "0\n"
            lines[idx+4] = shift

        if self.periodic == True and vdw == "MBD":
            idx = -1
            for i in range(len(lines)):
                if lines[i].find("KGRID") != -1:
                    idx = i
            lines[idx] = "    KGRID = "+str(mbdpacking[0])+" "+str(mbdpacking[1])+" "+str(mbdpacking[2])+"\n"
        if read_charges == True:
            idx = -1
            for i in range(len(lines)):
                if lines[i].find("ReadInitialCharges") != -1:
                    idx = i
            targetline = "  ReadInitialCharges = Yes\n"
            lines[idx] = targetline

            idx = -1
            for i in range(len(lines)):
                if lines[i].find("WriteChargesAsText") != -1:
                    idx = i
            targetline = "  ReadChargesAsText = Yes\n"
            lines[idx+1] = targetline

        with open(dir + 'DFTB+/dftb_in.hsd', 'w') as f:
            for i in range(len(lines)):
                f.write(lines[i])
                
        subprocess.run("./" + dir + "DFTB+/dftbOpt.sh", shell=True)

    def RunHessian(self,vdw=None):
        if vdw == None:
            shutil.copyfile('DFTB+/in_files/Hessian_calc.hsd', 'DFTB+/dftb_in.hsd')
        elif vdw == "MBD":
            shutil.copyfile('DFTB+/in_files/Hessian_calc_mbd.hsd', 'DFTB+/dftb_in.hsd')
        elif vdw == "PW":
            shutil.copyfile('DFTB+/in_files/Hessian_calc_pw.hsd', 'DFTB+/dftb_in.hsd')
        elif vdw == "TS":
            shutil.copyfile('DFTB+/in_files/Hessian_calc_ts.hsd', 'DFTB+/dftb_in.hsd')
        else:
            print ("Dispersion type not recognized")
            sys.exit()
        subprocess.run("./dftbOpt.sh", shell=True)

    def Displace(self,i,dv): #displaces atom i a dv vector distance (Bohr)
        self.x[i]=self.x[i]+dv[0]
        self.y[i]=self.y[i]+dv[1]
        self.z[i]=self.z[i]+dv[2]

    def Displace_UC(self,dv): #elongates an orthorombic unit cel (Bohr)
        if self.periodic == False:
            print ("System not periodic")
            sys.exit()
        self.unit_cell[1][0] = self.unit_cell[1][0] + dv[0]
        self.unit_cell[2][1] = self.unit_cell[2][1] + dv[1]
        self.unit_cell[3][2] = self.unit_cell[3][2] + dv[2]

    def Displace_UC_2(self,dv): #elongates the second vector of the unit cel (Bohr)
        if self.periodic == False:
            print ("System not periodic")
            sys.exit()
        self.unit_cell[2][0] = self.unit_cell[2][0] + dv[0]
        self.unit_cell[2][1] = self.unit_cell[2][1] + dv[1]
        self.unit_cell[2][2] = self.unit_cell[2][2] + dv[2]

    def GetVersor(self,i,j): #returns versor that points from i to j
        versor = [self.x[j]-self.x[i],self.y[j]-self.y[i],self.z[j]-self.z[i]]
        norm = np.sqrt((self.x[i]-self.x[j])**2+(self.y[i]-self.y[j])**2+(self.z[i]-self.z[j])**2)
        return np.array(versor)/norm

    def GetVector(self,i,j): #returns vector that points from i to j
        versor = [self.x[j]-self.x[i],self.y[j]-self.y[i],self.z[j]-self.z[i]]
        return np.array(versor)

    def PullBond(self,i,j,dv=0.001): #pulls atom j away from i a dv distance (Bohr)
        versor = self.GetVersor(i,j)
        versor = versor*dv

        self.x[j]=self.x[j]+versor[0]
        self.y[j]=self.y[j]+versor[1]
        self.z[j]=self.z[j]+versor[2]

    def RotateBond(self,i,j,k,rotvec,da=0.01): #rotates atom k towards j an angle da pivoting on i

        inner = np.inner(self.GetVector(i,k),self.GetVector(i,j)) / (LA.norm(self.GetVector(i,k))* LA.norm(self.GetVector(i,j)))
        rad_in = np.arccos(np.clip(inner, -1.0, 1.0))

        rotation_vector = da * rotvec
        rotation = R.from_rotvec(rotation_vector)
        v1 = rotation.apply(self.GetVector(i,k))

        inner = np.inner(v1,self.GetVector(i,j)) / (LA.norm(v1)* LA.norm(self.GetVector(i,j)))
        rad_out = np.arccos(np.clip(inner, -1.0, 1.0))

        if rad_out > rad_in:
            rotation_vector = -da * rotvec
            rotation = R.from_rotvec(rotation_vector)
            v1 = rotation.apply(self.GetVector(i,k))

        v1 = v1 + np.array([self.x[i],self.y[i],self.z[i]])
        self.x[k] = v1[0]
        self.y[k] = v1[1]
        self.z[k] = v1[2]

    def MoveBond(self,i,xyz,dv=0.001): #moves atom i in a given direction

        if xyz=="x":
            self.x[i]=self.x[i]+dv
        if xyz=="y":
            self.y[i]=self.y[i]+dv
        if xyz=="z":
            self.z[i]=self.z[i]+dv

    def MoveAll(self,dv): #moves all atoms in a given direction

        for i in range(self.Nat):
            self.x[i] = self.x[i] + dv[0]
            self.y[i] = self.y[i] + dv[1]
            self.z[i] = self.z[i] + dv[2]

    def SetDisplacements(self,dm): #moves all atoms using a displacement matrix

        for i in range(self.Nat):
            self.x[i]=self.x[i]+dm[i][0]
            self.y[i]=self.y[i]+dm[i][1]
            self.z[i]=self.z[i]+dm[i][2]

    def GetForces(self,dir=''): #load all of the total forces from DFTB
        try:
            file = open(dir + "DFTB+/detailed.out", "r+")
        except OSError:
            print ("Could not open detailed.out file")
            sys.exit()

        lines = file.readlines()
        forceindex = -1
        for i in range(len(lines)):
            if lines[i].find("Total Forces") != -1:
                forceindex = i
        lines = lines[forceindex+1:forceindex+self.Nat+1]

        Forces = []
        for i in range(len(lines)):
            a = lines[i].split(' ')
            a = list(filter(lambda x: x != '', a))
            # print(a)
            a = list(map(float, a[1:]))
            Forces.append(a)
        return np.array(Forces)

    def GetHessian(self,condensed=False): #load the hessian matrix from dftb
        try:
            file = open("DFTB+/hessian.out", "r+")
        except OSError:
            print ("Could not open detailed.out file")
            sys.exit()

        ceil = math.ceil(float(3*self.Nat)/float(4))
        lines = file.readlines()
        Hessian = []
        for i in range(int(len(lines)/ceil)):
            row = []
            for j in range(ceil):
                a = lines[i*ceil+j].split(' ')
                a = list(filter(lambda x: x != '', a))
                a = list(map(float, a))
                row = row + a
            Hessian.append(np.array(row))

        if condensed==True:
            HessianCond=[]
            for i in range(self.Nat*3):
                HessianCondColumn = []
                for j in range(self.Nat):
                    cond = np.sqrt(Hessian[i][j*3]**2+Hessian[i][j*3+1]**2+Hessian[i][j*3+2]**2)
                    HessianCondColumn.append(cond)
                HessianCond.append(np.array(HessianCondColumn))
            return np.array(HessianCond)
        return np.array(Hessian)

    def _condenseHessian(self,Hessian): #calculates the condensed Hessian for a given Hessian
        HessianCond=[]
        for i in range(self.Nat*3):
            HessianCondColumn = []
            for j in range(self.Nat):
                cond = np.sqrt(Hessian[i][j*3]**2+Hessian[i][j*3+1]**2+Hessian[i][j*3+2]**2)
                HessianCondColumn.append(cond)
            HessianCond.append(np.array(HessianCondColumn))
        return np.array(HessianCond)

    def GetEnergy(self, dir=''): #load the total energy from DFTB
        try:
            file = open(dir + "DFTB+/detailed.out", "r+")
        except OSError:
            print ("Could not open detailed.out file")
            sys.exit()

        lines = file.readlines()
        forceindex = -1
        for i in range(len(lines)):
            # print(i)
            if lines[i].find("Total energy:") != -1:
                forceindex = i
        lines = lines[forceindex]

        a = lines.split(' ')
        a = list(filter(lambda x: x != '', a))
        a = a[2]
        return float(a)
        

    def CalcBondedNeighbours(self,Cutoffneighbours): #calculates the list of bonds in the whole structure, bonds are defined using Cutoffneighbours

        self.bonds = []
        for i in range(self.Nat):
            bondidx = []
            distances = self.Distances(i)
            for k in range(self.Nat):
                if distances[k] <= Cutoffneighbours and k != i:
                    bondidx.append(k)
            self.bonds.append(bondidx)

    def CalcBondAngles(self): #calculates the list of bond angles in the whole structure, bonds are defined using Cutoffneighbours
        self.angles = []
        for i in range(self.Nat):
            p = permutations(self.bonds[i]) 
            p = list(p)
            if len(p) == 1:
                p = None
            else:  
                for k in range(len(p)):
                    p[k]=list(p[k][:2])
                for k in range(len(p)):
                    if k>=len(p):
                        break
                    for j in range(len(p)- 1, -1, -1):
                        if p[k][0] == p[j][0] and p[k][1] == p[j][1] and k!= j:
                            p.pop(j)
                    for j in range(len(p)- 1, -1, -1):
                        if p[k][0] == p[j][1] and p[k][1] == p[j][0] and k!= j:
                            p.pop(j)
            self.angles.append(p)

        for i in range(len(self.angles)): 
            if self.angles[i] != None:
                for k in range(len(self.angles[i])):
                    v1 = self.GetVersor(i,self.angles[i][k][0])
                    v2 = self.GetVersor(i,self.angles[i][k][1])
                    inner = np.inner(v1,v2)
                    rad = np.arccos(np.clip(inner, -1.0, 1.0))
                    self.angles[i][k].append(rad)
                    cross = np.cross(v1,v2)
                    if rad <= 3.14159265359 and rad > 3.14159265358:
                        inner = np.inner(v1,v2+np.array([0,0,0.001]))
                        rad = np.arccos(np.clip(inner, -1.0, 1.0))
                        cross = np.cross(v1,v2+np.array([0,0,0.001]))
                        if rad <= rad <= 3.14159265359 and rad > 3.14159265358:
                            inner = np.inner(v1,v2+np.array([0,0.001,0]))
                            rad = np.arccos(np.clip(inner, -1.0, 1.0))
                            cross = np.cross(v1,v2+np.array([0,0.001,0]))

                    cross = cross/LA.norm(cross)
                    self.angles[i][k].append(cross)


    def CalcBondOffPlane(self): #calculates the list of offplane angles in the whole structure, bonds are defined using Cutoffneighbours
        self.offplane = []
        for i in range(self.Nat):

            if len(self.bonds[i])>2:
                self.offplane.append(self.bonds[i][:3])
            else:
                self.offplane.append(None)

        for i in range(len(self.offplane)): 
            if self.offplane[i] != None:
                v1 = self.GetVersor(i,self.offplane[i][0])
                v2 = self.GetVersor(i,self.offplane[i][1])
                v3 = self.GetVersor(i,self.offplane[i][2])
                v = np.cross(v1,v2) + np.cross(v2,v3) + np.cross(v3,v1)
                h = np.inner(v/LA.norm(v), v1)
                self.offplane[i].append(np.arcsin(h))
                # v1 = self.GetVersor(self.offplane[i][0],self.offplane[i][1])
                # v2 = self.GetVersor(self.offplane[i][0],self.offplane[i][2])
                # cross = np.cross(v1,v2)
                # cross = cross/LA.norm(cross)
                # v = self.GetVector(i,self.offplane[i][1])
                # distance = LA.norm(v)
                # inner = np.inner(v,cross)
                # self.offplane[i].append(np.arcsin(inner/distance))

    def CalcBondDihedral(self): #calculates the list of dihedral angles in the whole structure, bonds are defined using Cutoffneighbours
        self.dihedral = []
        for i in range(self.Nat):
            for j in self.bonds[i]:
                for k in self.bonds[j]:
                    for l in self.bonds[k]:
                        if (j in self.bonds[i]) and (k in self.bonds[j]) and (k not in self.bonds[i]) and (l in self.bonds[k]) and (l not in self.bonds[j]) and (l not in self.bonds[i]):
                            rij = self.GetVector(i,j)
                            rjk = self.GetVector(j,k)
                            rlk = self.GetVector(l,k)
                            m = np.cross(rij,rjk)
                            n = np.cross(rlk,rjk)
                            sin = np.inner(n,rij)*LA.norm(rjk)/(LA.norm(m)*LA.norm(n))
                            alreadyconsidered = False
                            for it in range(len(self.dihedral)):
                                if self.dihedral[it][:4] == [l,k,j,i]:
                                    alreadyconsidered = True
                                    break
                            if alreadyconsidered == False:
                                self.dihedral.append([i,j,k,l,np.arcsin(sin)])

    def CalcBondDistances(self): #calculates the list of bond angles in the whole structure, bonds are defined using Cutoffneighbours
        self.distances = []
        for i in range(self.Nat):
            row = []
            for j in range(self.Nat):
                row.append(self.Distance(i,j))
            self.distances.append(row)


                

