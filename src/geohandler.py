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

from pathlib import Path
current_file_path = Path(__file__).resolve()
current_dir = str(current_file_path.parent.parent)

class Handler():
    """
    A class to handle geometric operations on molecular structures.

    Attributes:
    Nat (int): Number of atoms in the structure.
    R0 (float): Reference interatomic distance.
    R0s (np.ndarray): Array of interatomic distances.
    R0neighbours (int): Number of nearest neighbors to consider for R0 approximation.
    bonds (list): List of bonded atom pairs.
    offplane (list): List of off-plane angles.
    dihedral (list): List of dihedral angles.
    angles (list): List of bond angles.
    distances (list): List of interatomic distances.
    periodic (bool): Indicates if the structure is periodic.
    types (list): List of atom types.
    unit_cell (list): List of unit cell vectors.

    Methods:
    add: Adds two structures together.
    Fold: Folds a periodic structure into its unit cell.
    Expand: Expands the unit cell of a periodic structure.
    RemoveAtoms: Removes atoms from the structure.
    SetPos: Abstract method to set the positions of atoms.
    Pos: Returns the positions of atoms.
    PosAsList: Returns the positions of atoms as a list.
    PosAsListIdx: Returns the positions of atoms with given indices as a list.
    TypesAsListIdx: Returns the types of atoms with given indices as a list.
    ShowStruct: Displays the 3D structure.
    ShowR0s: Displays the R0 estimations.
    ShowWidths: Displays the errors in the R0 estimations.
    ShowDistances: Displays the interatomic distances for a given atom.
    SaveR0s: Saves the R0 estimations to a file.
    Distances: Returns a list of all interatomic distances from a given atom.
    Distance: Returns the distance between two atoms.
    GetVersor: Returns the versor pointing from one atom to another.
    SaveDistances: Saves all interatomic distances to a file.
    GetR0s: Returns a list of R0 estimations and errors.
    UpdateR0s: Updates the R0 estimations and errors.
    SaveGeometry: Saves the geometry to a file.
    LoadGeometry: Loads the geometry from a file.
    RunOptimize: Runs a DFTB+ optimization.
    RunStatic: Runs a static DFTB+ calculation.
    RunHessian: Runs a DFTB+ Hessian calculation.
    Displace: Displaces an atom by a given vector.
    SetAtomPos: Sets the position of an atom.
    Displace_UC: Elongates the unit cell.
    Displace_UC_2: Elongates the second vector of the unit cell.
    GetVersor: Returns a versor pointing from one atom to another.
    PullBond: Pulls one atom away from another.
    RotateBond: Rotates one atom towards another around a pivot atom.
    RotateAllZ: Rotates all atoms around the Z-axis.
    MoveBond: Moves an atom in a given direction.
    MoveAll: Moves all atoms in a given direction.
    Center: Centers the structure.
    SetDisplacements: Moves all atoms using a displacement matrix.
    GetForces: Loads the forces from a DFTB+ calculation.
    GetVolume: Loads the CPA ratio from a DFTB+ calculation.
    GetStress: Loads the stress tensor from a DFTB+ calculation.
    GetHessian: Loads the Hessian matrix from a DFTB+ calculation.
    _condenseHessian: Calculates the condensed Hessian matrix.
    GetEnergy: Loads the total energy from a DFTB+ calculation.
    CalcBondedNeighbours: Calculates bonded neighbours based on a cutoff distance.
    CalcBondAngles: Calculates bond angles.
    CalcBondOffPlane: Calculates off-plane angles.
    CalcBondDihedral: Calculates dihedral angles.
    CalcBondDistances: Calculates bond distances.
    ConectedComponents: Returns all connected components in the structure.
    """
    def __init__(self,Nat,R0,propcalc=True): #initialize the desired geometry with interatomic distance R0 (Bohr)
        self.Nat = Nat
        self.R0s = None
        self.R0 = R0 #interatomic distance in bohr
        self.widths = None
        self.R0neighbours = 2 #number of closest neighbours to calculate the R0 approximation
        if self.Nat <= self.R0neighbours:
            self.R0neighbours=self.Nat-1
        self.SetPos(self.Nat,self.R0)
        self.Cutoffneighbours = self.R0*1.5 #cutoffdistance for bonded neighbours
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
        self.types = False
        self.unit_cell = []

    def add(self,structure,propcalc=False): #adds two structures together, the added structure indexes are concatened after the original
        self.Nat = self.Nat+structure.Nat
        # self.x = self.x.tolist()
        # self.y = self.y.tolist()
        # self.z = self.z.tolist()
        for i in range(structure.Nat):
            self.x.append(structure.x[i])
            self.y.append(structure.y[i])
            self.z.append(structure.z[i])
        if propcalc==True:
            self.CalcBondedNeighbours(self.Cutoffneighbours)
            self.CalcBondDistances()
            self.CalcBondAngles()
            self.CalcBondOffPlane()
            self.CalcBondDihedral()
        print(self.types)
        self.types = self.types + structure.types

    def Fold(self): #folds an orthorombic periodic structure into its unit-cell following its periodicity
        if self.periodic == False:
            print("structure must be periodic in order to fold onto unit cell...")
            return
        for i in range(self.Nat):
            if int(self.x[i] // self.unit_cell[1][0]) != 0:
                self.x[i] = self.x[i] - int(self.x[i] // self.unit_cell[1][0])*self.unit_cell[1][0]

            if int(self.y[i] // self.unit_cell[2][1]) != 0:
                self.y[i] = self.y[i] - int(self.y[i] // self.unit_cell[2][1])*self.unit_cell[2][1]

            if int(self.z[i] // self.unit_cell[3][2]) != 0:
                self.z[i] = self.z[i] - int(self.z[i] // self.unit_cell[3][2])*self.unit_cell[3][2]


    def Expand(self,exp_vec=[2,2,2],trim = None): #expands the unit cell of a periodic structure following its periodicity. trim will cut away all atoms further than the unit cell lenght*trim
        Nat_init = self.Nat
        if self.periodic == False:
            print("structure must be periodic in order to expand...")
            return

        if trim == None:
            for j in range(1,exp_vec[0]):
                for i in range(self.Nat):
                    self.x.append(self.x[i]+j*self.unit_cell[1][0])
                    self.y.append(self.y[i])
                    self.z.append(self.z[i])
                self.types = self.types + self.types
                for i in range(self.Nat):
                    self.x.append(self.x[i]-j*self.unit_cell[1][0])
                    self.y.append(self.y[i])
                    self.z.append(self.z[i])
                self.types = self.types + self.types
            self.Nat = self.Nat*(2*exp_vec[0]-1)

            for j in range(1,exp_vec[1]):
                for i in range(self.Nat):
                    self.x.append(self.x[i])
                    self.y.append(self.y[i]+j*self.unit_cell[2][1])
                    self.z.append(self.z[i])
                self.types = self.types + self.types
                for i in range(self.Nat):
                    self.x.append(self.x[i])
                    self.y.append(self.y[i]-j*self.unit_cell[2][1])
                    self.z.append(self.z[i])
                self.types = self.types + self.types
            self.Nat = self.Nat*(2*exp_vec[1]-1)

            for j in range(1,exp_vec[2]):
                for i in range(self.Nat):
                    self.x.append(self.x[i])
                    self.y.append(self.y[i])
                    self.z.append(self.z[i]+j*self.unit_cell[3][2])
                self.types = self.types + self.types
                for i in range(self.Nat):
                    self.x.append(self.x[i])
                    self.y.append(self.y[i])
                    self.z.append(self.z[i]-j*self.unit_cell[3][2])
                self.types = self.types + self.types
            self.Nat = self.Nat*(2*exp_vec[2]-1)
        else:
            bound_delta = self.unit_cell[1][0]*(trim-1)
            top = bound_delta +  self.unit_cell[1][0]
            bottom = -bound_delta
            for j in range(1,exp_vec[0]):
                for i in range(self.Nat):
                    if self.x[i]+j*self.unit_cell[1][0] < top and self.x[i]+j*self.unit_cell[1][0] >= self.unit_cell[1][0]: 
                        self.x.append(self.x[i]+j*self.unit_cell[1][0])
                        self.y.append(self.y[i])
                        self.z.append(self.z[i])
                        self.types.append(self.types[i])
                        self.Nat = self.Nat + 1
                for i in range(self.Nat):
                    if self.x[i]-j*self.unit_cell[1][0] > bottom and self.x[i]-j*self.unit_cell[1][0] < 0: 
                        self.x.append(self.x[i]-j*self.unit_cell[1][0])
                        self.y.append(self.y[i])
                        self.z.append(self.z[i])
                        self.types.append(self.types[i])
                        self.Nat = self.Nat + 1

            bound_delta = self.unit_cell[2][1]*(trim-1)
            top = bound_delta +  self.unit_cell[2][1]
            bottom = -bound_delta
            for j in range(1,exp_vec[1]):
                for i in range(self.Nat):
                    if self.y[i]+j*self.unit_cell[2][1] < top and self.y[i]+j*self.unit_cell[2][1] >= self.unit_cell[2][1]: 
                        self.x.append(self.x[i])
                        self.y.append(self.y[i]+j*self.unit_cell[2][1])
                        self.z.append(self.z[i])
                        self.types.append(self.types[i])
                        self.Nat = self.Nat + 1
                for i in range(self.Nat):
                    if self.y[i]-j*self.unit_cell[2][1] > bottom and self.y[i]-j*self.unit_cell[2][1] < 0: 
                        self.x.append(self.x[i])
                        self.y.append(self.y[i]-j*self.unit_cell[2][1])
                        self.z.append(self.z[i])
                        self.types.append(self.types[i])
                        self.Nat = self.Nat + 1

            bound_delta = self.unit_cell[3][2]*(trim-1)
            top = bound_delta +  self.unit_cell[3][2]
            bottom = -bound_delta
            for j in range(1,exp_vec[2]):
                for i in range(self.Nat):
                    if self.z[i]+j*self.unit_cell[3][2] < top and self.z[i]+j*self.unit_cell[3][2] >= self.unit_cell[3][2]: 
                        self.x.append(self.x[i])
                        self.y.append(self.y[i])
                        self.z.append(self.z[i]+j*self.unit_cell[3][2])
                        self.types.append(self.types[i])
                        self.Nat = self.Nat + 1
                for i in range(self.Nat):
                    if self.z[i]-j*self.unit_cell[3][2] > bottom and self.z[i]-j*self.unit_cell[3][2] < 0: 
                        self.x.append(self.x[i])
                        self.y.append(self.y[i])
                        self.z.append(self.z[i]-j*self.unit_cell[3][2])
                        self.types.append(self.types[i])
                        self.Nat = self.Nat + 1

        print("structure expanded "+str(self.Nat/Nat_init)+" times..")
        self.pos_as_nparray = np.array(self.PosAsList())

    def RemoveAtoms(self,idx): #removes all atoms in the list idx from the geometry
        for index in sorted(idx, reverse=True):
            del self.x[index]
            del self.y[index]
            del self.z[index]
            del self.types[index]
        self.Nat = self.Nat - len(idx)

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
    def PosAsListIdx(self,idx): #returns the positions of every atom in the list idx as a list
        pos = []
        for i in idx:
            pos.append([self.x[i],self.y[i],self.z[i]])
        return pos
    def TypesAsListIdx(self,idx): #returns the types of every atom in the list idx as a list
        types = []
        for i in idx:
            types.append(self.types[i])
        return types

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

        ax.scatter(self.x,self.y,self.z);
        # ax.set_ylim((-10,10))
        # ax.set_xlim((-10,10))
        # ax.set_zlim((-10,10))
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
        with open(current_dir+'/out/R0s.txt', 'w') as f:
            for i in range(len(self.R0s)):
                f.write(str(i)+' '+str(self.R0s[i])+'\n')

    def Distances(self,idx): #returns a list of all the interatomic distances from atom idx
        ref_point = np.array(self.pos_as_nparray[idx])
        diff = (self.pos_as_nparray - ref_point)**2
        distances = np.sqrt(np.sum(diff,axis=1))
        return np.array(distances)

    def Distance(self,i,j): #returns the distance between atom i and j
        return np.sqrt((self.x[i]-self.x[j])**2+(self.y[i]-self.y[j])**2+(self.z[i]-self.z[j])**2)

    def GetVersor(self,i,j): #returns the versor pointing from atom i to j
        v = [(self.x[j]-self.x[i])/self.Distance(i,j),(self.y[j]-self.y[i])/self.Distance(i,j),(self.z[j]-self.z[i])/self.Distance(i,j)]
        return np.array(v)

    def SaveDistances(self): #Saves all the distances (Bohr) for each atom to a file
        dist = []
        for i in range(self.Nat):
            distances = self.Distances(i)
            distances.sort()
            dist.append(distances)

        print("Saving distances..")
        with open(current_dir+'/out/Distances.txt', 'w') as f:
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

    def SaveGeometry(self,decour="geom",path=None,instruct=False,charges=None,decour_charges="charges"): #saves the geometry to a gen file in angstroms
        print("Saving geometry..")
        angstrom = 0.529177249
        folder = current_dir+"/DFTB+"
        multiatom = False
        if instruct == True:
            folder = current_dir+"/SavedStructures"
        if self.periodic == False:
            name=folder+'/'+decour+'.gen'
            if path != None:
                name = folder+'/'+path+'/'+decour+'.gen'
                if not os.path.exists(folder+'/'+path):
                    os.makedirs(folder+'/'+path)


            with open(name, 'w') as f:
                f.write(str(self.Nat)+' C\n')

                types = []
                types.append(self.types[0])
                for i in range(1,self.Nat):
                    if self.types[i] not in types:
                        types.append(self.types[i])

                header = '  '
                for i in range(len(types)):
                    header=header+types[i]+' '
                header = header+'\n'
                f.write(header)
                print("Atom types:"+header)
                for i in range(self.Nat):
                    for j in range(len(types)):
                        if self.types[i] == types[j]:
                            f.write('  '+str(i+1)+' '+str(j+1)+'  '+str(angstrom*self.x[i])+' '+str(angstrom*self.y[i])+' '+str(angstrom*self.z[i])+'\n')

        elif self.periodic == True:
            name=folder+'/'+decour+'.gen'
            if path != None:
                name = folder+'/'+path+'/'+decour+'.gen'
                if not os.path.exists(folder+'/'+path):
                    os.makedirs(folder+'/'+path)
            with open(name, 'w') as f:
                f.write(str(self.Nat)+' S\n')
                f.write('  C H\n')
                for i in range(self.Nat):
                    if self.types[i] == "C":
                        f.write('  '+str(i+1)+' 1  '+str(angstrom*self.x[i])+' '+str(angstrom*self.y[i])+' '+str(angstrom*self.z[i])+'\n')
                    elif self.types[i] == "H":
                        f.write('  '+str(i+1)+' 2  '+str(angstrom*self.x[i])+' '+str(angstrom*self.y[i])+' '+str(angstrom*self.z[i])+'\n')
                for i in range(len(self.unit_cell)):
                    f.write('  '+str(angstrom*self.unit_cell[i][0])+' '+str(angstrom*self.unit_cell[i][1])+' '+str(angstrom*self.unit_cell[i][2])+'\n')
        if charges != None:
            totalcharge = 0
            for i in range(self.Nat):
                totalcharge=totalcharge+charges[i][6]
            with open("../DFTB+/charges.dat", 'w') as f:
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



    def LoadGeometry(self,path="geom.out.gen"): #loads the geometry from a gen, xyz or sdf file in angstroms and converts it into Bohr to be used internally
        print("Loading geometry..")
        angstrom = 0.529177249
        nantobohr= 0.0529177
        extension = path[-3:]
        recognized = False
        self.periodic = False

        if extension == "sdf":
            recognized = True
            if path=="Graphene-C92.sdf":
                angstrom = angstrom * 0.71121438
            self.Nat, geometry = filetypes.Loadsdf(current_dir+"/SavedStructures/"+path,angstrom)
            print(str(self.Nat)+" atoms loaded")
                
        if extension == "gen":
            recognized = True
            if path != "geom.out.gen" and path != "geo_end.gen":
                self.Nat, geometry, self.periodic, self.types, self.unit_cell = filetypes.Loadgen(current_dir+"/SavedStructures/"+path,angstrom)
                print(str(self.Nat)+" atoms loaded")
            else:
                self.Nat, geometry, self.periodic, self.types, self.unit_cell = filetypes.Loadgen(current_dir+"/DFTB+/"+path,angstrom)

        if extension == "gro":
            recognized = True
            self.Nat, geometry, self.periodic, self.types, self.unit_cell = filetypes.Loadgro(current_dir+"/SavedStructures/"+path,nantobohr)
            print(str(self.Nat)+" atoms loaded")

        if extension == "xyz":
            recognized = True
            self.Nat, geometry = filetypes.Loadxyz_single(current_dir+"/SavedStructures/"+path,angstrom)

        if extension == "cc1":
            recognized = True
            self.Nat, geometry, self.types = filetypes.Loadcc1(current_dir+"/SavedStructures/"+path,angstrom)

        if extension == "txt":
            recognized = True
            self.Nat, geometry = filetypes.Loadtxt(current_dir+"/SavedStructures/"+path,angstrom)
            print(str(self.Nat)+" atoms loaded")

        if extension == "ply":
            recognized = True
            self.Nat, geometry = filetypes.LoadPly(current_dir+"/SavedStructures/"+path,10)
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

    def RunOptimize(self,vdw=None,static=None,read_charges=False,packing=[5,5,5],mbdpacking=[3,3,3], fixlengths=False): #Runs a DFTB+ optimization
        if self.periodic == False:
            if vdw == None:
                shutil.copyfile(current_dir+'/DFTB+/in_files/optimize.hsd', current_dir+'/DFTB+/dftb_in.hsd')
            elif vdw == "MBD":
                shutil.copyfile(current_dir+'/DFTB+/in_files/optimize_mbd.hsd', current_dir+'/DFTB+/dftb_in.hsd')
            elif vdw == "PW":
                shutil.copyfile(current_dir+'/DFTB+/in_files/optimize_pw.hsd', current_dir+'/DFTB+/dftb_in.hsd')
            elif vdw == "TS":
                shutil.copyfile(current_dir+'/DFTB+/in_files/optimize_ts.hsd', current_dir+'/DFTB+/dftb_in.hsd')
            else:
                print ("Dispersion type not recognized")
                sys.exit()
            try:
                file = open("../DFTB+/dftb_in.hsd", "r+")
            except OSError:
                print ("Could not open dftb_in.hsd file")
                sys.exit()
        else:
            if vdw == None:
                shutil.copyfile(current_dir+'/DFTB+/in_files/optimize-periodic.hsd', current_dir+'/DFTB+/dftb_in.hsd')
            elif vdw == "MBD":
                shutil.copyfile(current_dir+'/DFTB+/in_files/optimize-periodic_mbd.hsd', current_dir+'/DFTB+/dftb_in.hsd')
            elif vdw == "PW":
                shutil.copyfile(current_dir+'/DFTB+/in_files/optimize-periodic_pw.hsd', current_dir+'/DFTB+/dftb_in.hsd')
            elif vdw == "TS":
                shutil.copyfile(current_dir+'/DFTB+/in_files/optimize-periodic_ts.hsd', current_dir+'/DFTB+/dftb_in.hsd')
            else:
                print ("Dispersion type not recognized")
                sys.exit()
            try:
                file = open(current_dir+"/DFTB+/dftb_in.hsd", "r+")
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
            if fixlengths == "X":
                lines[idx] = "  FixLengths = {Yes No No}"+"\n"
            if fixlengths == "Y":
                lines[idx] = "  FixLengths = {No Yes No}"+"\n"
            if fixlengths == "Z":
                lines[idx] = "  FixLengths = {No No Yes}"+"\n"
            if fixlengths == "XYZ":
                lines[idx] = "  FixLengths = {Yes Yes Yes}"+"\n"
            if fixlengths != "X" and fixlengths != "Y" and fixlengths != "Z" and fixlengths != "XYZ":
                print("Wrong imput for fixlengths (should be 'X', 'Y', 'Z' or 'XYZ')")
                sys.exit()
        if static != None:
            idx = -1
            for i in range(len(lines)):
                if lines[i].find("MovedAtoms") != -1:
                    idx = i
            targetline = "  MovedAtoms = !("
            for j in range(len(static)):
                targetline = targetline + str(static[j]+1) + " "
            targetline = targetline + ")"+"\n"
            lines[idx] = targetline

        multiatom = False

        for i in range(len(self.types)):
            if self.types[i] == "H":
                multiatom = True
        print("Multiatom:",multiatom)
        if multiatom == True:
            idx = -1
            for i in range(len(lines)):
                if lines[i].find("SlaterKosterFiles") != -1:
                    idx = i
            lines.insert(idx+1,'    H-H = "../../Slater-Koster/3ob-3-1/H-H.skf"\n')
            lines.insert(idx+1,'    C-H = "../../Slater-Koster/3ob-3-1/C-H.skf"\n')
            lines.insert(idx+1,'    H-C = "../../Slater-Koster/3ob-3-1/H-C.skf"\n')
            for i in range(len(lines)):
                if lines[i].find("HubbardDerivs") != -1:
                    idx = i
            lines.insert(idx+1,'    H = -0.1857\n')

            for i in range(len(lines)):
                if lines[i].find("MaxAngularMomentum") != -1:
                    idx = i
            lines.insert(idx+1,'    H = "p"\n')

        types = []
        types.append(self.types[0])
        for i in range(1,self.Nat):
            if self.types[i] not in types:
                types.append(self.types[i])
        if "Si" in types:
            idx = -1
            for i in range(len(lines)):
                if lines[i].find("ThirdOrderFull") != -1:
                    lines[i]="  ThirdOrderFull = No\n"
            for i in range(len(lines)):
                if lines[i].find("HubbardDerivs") != -1:
                    idx = i
            lines.pop(idx+2)
            lines.pop(idx+1)
            lines.pop(idx)
            idx = -1
            for i in range(len(lines)):
                if lines[i].find("SlaterKosterFiles") != -1:
                    lines[i+1] = '    Si-Si = "../../Slater-Koster/matsci-0-3/Si-Si.skf"\n'
                if lines[i].find("MaxAngularMomentum") != -1:
                    lines[i+1] = '    Si = "p"\n'

        with open(current_dir+'/DFTB+/dftb_in.hsd', 'w') as f:
            for i in range(len(lines)):
                f.write(lines[i])
        subprocess.run(current_dir+"/src/dftbOpt.sh", shell=True)

    def RunStatic(self,vdw=None,read_charges=False,packing=[5,5,5],mbdpacking=[2,2,2],get_CPA=False):
        if self.periodic == False:
            if vdw == None:
                shutil.copyfile(current_dir+'/DFTB+/in_files/static_calc.hsd', current_dir+'/DFTB+/dftb_in.hsd')
            elif vdw == "MBD":
                shutil.copyfile(current_dir+'/DFTB+/in_files/static_calc_mbd.hsd', current_dir+'/DFTB+/dftb_in.hsd')
            elif vdw == "PW":
                shutil.copyfile(current_dir+'/DFTB+/in_files/static_calc_pw.hsd', current_dir+'/DFTB+/dftb_in.hsd')
            elif vdw == "TS":
                shutil.copyfile(current_dir+'/DFTB+/in_files/static_calc_ts.hsd', current_dir+'/DFTB+/dftb_in.hsd')
            else:
                print ("Dispersion type not recognized")
                sys.exit()
        else:
            if vdw == None:
                shutil.copyfile(current_dir+'/DFTB+/in_files/static_calc-periodic.hsd', current_dir+'/DFTB+/dftb_in.hsd')
            elif vdw == "MBD":
                shutil.copyfile(current_dir+'/DFTB+/in_files/static_calc-periodic_mbd.hsd', current_dir+'/DFTB+/dftb_in.hsd')
            elif vdw == "PW":
                shutil.copyfile(current_dir+'/DFTB+/in_files/static_calc-periodic_pw.hsd', current_dir+'/DFTB+/dftb_in.hsd')
            elif vdw == "TS":
                shutil.copyfile(current_dir+'/DFTB+/in_files/static_calc-periodic_ts.hsd', current_dir+'/DFTB+/dftb_in.hsd')
            else:
                print ("Dispersion type not recognized")
                sys.exit()

        try:
            file = open(current_dir+"/DFTB+/dftb_in.hsd", "r+")
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


        multiatom = False
        
        for i in range(len(self.types)):
            if self.types[i] == "H":
                multiatom = True
        print("Multiatom:",multiatom)
        if multiatom == True:
            idx = -1
            for i in range(len(lines)):
                if lines[i].find("SlaterKosterFiles") != -1:
                    idx = i
            lines.insert(idx+1,'    H-H = "../../Slater-Koster/3ob-3-1/H-H.skf"\n')
            lines.insert(idx+1,'    C-H = "../../Slater-Koster/3ob-3-1/C-H.skf"\n')
            lines.insert(idx+1,'    H-C = "../../Slater-Koster/3ob-3-1/H-C.skf"\n')
            for i in range(len(lines)):
                if lines[i].find("HubbardDerivs") != -1:
                    idx = i
            lines.insert(idx+1,'    H = -0.1857\n')

            for i in range(len(lines)):
                if lines[i].find("MaxAngularMomentum") != -1:
                    idx = i
            lines.insert(idx+1,'    H = "p"\n')

        types = []
        types.append(self.types[0])
        for i in range(1,self.Nat):
            if self.types[i] not in types:
                types.append(self.types[i])
        if "Si" in types:
            idx = -1
            for i in range(len(lines)):
                if lines[i].find("ThirdOrderFull") != -1:
                    lines[i]="  ThirdOrderFull = No\n"
            for i in range(len(lines)):
                if lines[i].find("HubbardDerivs") != -1:
                    idx = i
            lines.pop(idx+2)
            lines.pop(idx+1)
            lines.pop(idx)
            idx = -1
            for i in range(len(lines)):
                if lines[i].find("SlaterKosterFiles") != -1:
                    lines[i+1] = '    Si-Si = "../../Slater-Koster/matsci-0-3/Si-Si.skf"\n'
                if lines[i].find("MaxAngularMomentum") != -1:
                    lines[i+1] = '    Si = "p"\n'
        if get_CPA == True:
            for i in range(len(lines)):
                if lines[i].find("Options {") != -1:
                    lines.insert(i+1,'  WriteCPA = Yes  \n')
                
        with open(current_dir+'/DFTB+/dftb_in.hsd', 'w') as f:
            for i in range(len(lines)):
                f.write(lines[i])


        subprocess.run(current_dir+"/src/dftbOpt.sh", shell=True)

    def RunHessian(self,vdw=None):
        if vdw == None:
            shutil.copyfile(current_dir+'/DFTB+/in_files/Hessian_calc.hsd', current_dir+'/DFTB+/dftb_in.hsd')
        elif vdw == "MBD":
            shutil.copyfile(current_dir+'/DFTB+/in_files/Hessian_calc_mbd.hsd', current_dir+'/DFTB+/dftb_in.hsd')
        elif vdw == "PW":
            shutil.copyfile(current_dir+'/DFTB+/in_files/Hessian_calc_pw.hsd', current_dir+'/DFTB+/dftb_in.hsd')
        elif vdw == "TS":
            shutil.copyfile(current_dir+'/DFTB+/in_files/Hessian_calc_ts.hsd', current_dir+'/DFTB+/dftb_in.hsd')
        else:
            print ("Dispersion type not recognized")
            sys.exit()
        subprocess.run(current_dir+"/src/dftbOpt.sh", shell=True)

    def Displace(self,i,dv): #displaces atom i a dv vector distance (Bohr)
        self.x[i]=self.x[i]+dv[0]
        self.y[i]=self.y[i]+dv[1]
        self.z[i]=self.z[i]+dv[2]

    def SetAtomPos(self,i,v): #sets the position of atom i at v (Bohr)
        self.x[i]=v[0]
        self.y[i]=v[1]
        self.z[i]=v[2]

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
        
    def RotateAllZ(self,angle):
        for i in range(self.Nat):
            aux = self.x[i]*np.cos(angle)- self.y[i]*np.sin(angle)
            self.y[i] = self.x[i]*np.sin(angle)+ self.y[i]*np.cos(angle)
            self.x[i] = aux


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

    def Center(self): #centers the structure 
        pos = self.PosAsList()
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
        self.MoveAll([-x_med,-y_med,-z_med])

    def SetDisplacements(self,dm): #moves all atoms using a displacement matrix

        for i in range(self.Nat):
            self.x[i]=self.x[i]+dm[i][0]
            self.y[i]=self.y[i]+dm[i][1]
            self.z[i]=self.z[i]+dm[i][2]

    def GetForces(self): #load all of the total forces from DFTB
        try:
            file = open(current_dir+"/DFTB+/detailed.out", "r+")
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
            a = list(map(float, a[1:]))
            Forces.append(a)
        return np.array(Forces)

    def GetVolume(self,idx): #loads the CPA ratio from atom idx
        try:
            file = open(current_dir+"/DFTB+/CPA_ratios.out", "r+")
        except OSError:
            print ("Could not CPA_ratios file")
            sys.exit()
        lines = file.readlines()

        volume = lines[idx+1].split(' ')
        volume = list(filter(lambda x: x != '', volume))
        volume = float(volume[idx+1])
        return volume

    def GetStress(self): #load all of the total forces from DFTB
        try:
            file = open(current_dir+"/DFTB+/detailed.out", "r+")
        except OSError:
            print ("Could not open detailed.out file")
            sys.exit()

        lines = file.readlines()
        forceindex = -1
        for i in range(len(lines)):
            if lines[i].find("Total stress tensor") != -1:
                forceindex = i
        lines = lines[forceindex+1:forceindex+1+3]

        Forces = []
        for i in range(len(lines)):
            a = lines[i].split(' ')
            a = list(filter(lambda x: x != '', a))
            a = list(map(float, a[0:]))
            Forces.append(a)
        return np.array(Forces)

    def GetHessian(self,condensed=False): #load the hessian matrix from dftb
        try:
            file = open(current_dir+"/DFTB+/hessian.out", "r+")
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

    def GetEnergy(self): #load the total energy from DFTB
        try:
            file = open(current_dir+"/DFTB+/detailed.out", "r+")
        except OSError:
            print ("Could not open detailed.out file")
            sys.exit()

        lines = file.readlines()
        forceindex = -1
        for i in range(len(lines)):
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

    def ConectedComponents(self): #returns all of the connected components in a graph determined by the bonds in the structure
        self.CalcBondedNeighbours(self.R0*1.3)
        graph = self.bonds
        visited = set()
        components = []
        def dfs(node):
            visited.add(node)
            component.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor)
        
        for node in range(len(graph)):
            if node not in visited:
                component = []
                dfs(node)
                components.append(component)
        
        return components
                

