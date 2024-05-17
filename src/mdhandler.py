import copy
import numpy as np
import shutil
import subprocess
import filetypes
import sys
import structures


from pathlib import Path
current_file_path = Path(__file__).resolve()
current_dir = str(current_file_path.parent.parent)

class Handler():

    def __init__(self,structure,optimize=True):
        self.structure_eq = copy.deepcopy(structure)

        if optimize == True:
            self.structure_eq.SaveGeometry()
            self.structure_eq.RunOptimize()
            self.structure_eq.LoadGeometry()
            self.structure_eq.SaveGeometry()
        else:
            self.structure_eq.SaveGeometry()
        self.evolution = None
        self.acelerations = None #acceleration in a.u.
        self.types = self.structure_eq.types

    def RunMD(self,steps,temp=400,vdw=None,keepstationary=False,static=None,save_steps=1):
        if vdw == None:
            shutil.copyfile(current_dir+'/DFTB+/in_files/md.hsd', current_dir+'/DFTB+/dftb_in.hsd')
        elif vdw == "MBD":
            shutil.copyfile(current_dir+'/DFTB+/in_files/md_mbd.hsd', current_dir+'/DFTB+/dftb_in.hsd')
        elif vdw == "PW":
            shutil.copyfile(current_dir+'/DFTB+/in_files/md_pw.hsd', current_dir+'/DFTB+/dftb_in.hsd')
        elif vdw == "TS":
            shutil.copyfile(current_dir+'/DFTB+/in_files/md_ts.hsd', current_dir+'/DFTB+/dftb_in.hsd')
        else:
            print ("Dispersion type not recognized")
            sys.exit()
        try:
            file = open(current_dir+"/DFTB+/dftb_in.hsd", "r+")
        except OSError:
            print ("Could not open detailed.out file")
            sys.exit()

        lines = file.readlines()
        file.close()
        idx = -1
        for i in range(len(lines)):
            if lines[i].find("Steps") != -1:
                idx = i
        targetline = "  Steps = " + str(steps) +"\n"
        lines[idx] = targetline

        idx = -1
        for i in range(len(lines)):
            if lines[i].find("MDRestartFrequency") != -1:
                idx = i
        targetline = "  MDRestartFrequency = " + str(save_steps) +"\n"
        lines[idx] = targetline

        if keepstationary == False:
            keepstationary = "No"
        else:
            keepstationary = "Yes"
        idx = -1
        for i in range(len(lines)):
            if lines[i].find("KeepStationary") != -1:
                idx = i
        targetline = "  KeepStationary = " + keepstationary +"\n"
        lines[idx] = targetline

        if temp > 0:
            idx = -1
            for i in range(len(lines)):
                if lines[i].find("Temperature [Kelvin]") != -1:
                    idx = i
            targetline = "    Temperature [Kelvin] = " + str(temp) +"\n"
            lines[idx] = targetline
        else:
            idx = -1
            for i in range(len(lines)):
                if lines[i].find("  Thermostat =") != -1:
                    idx = i
            targetline = "  Thermostat = None{" + "\n"
            lines[idx] = targetline
            lines[idx+1] = "    InitialTemperature = 0" + "\n"
            lines.pop(idx+2)
            

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

        for i in range(len(self.structure_eq.types)):
            if self.structure_eq.types[i] == "H":
                multiatom = True
        print(multiatom)
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
            

        with open(current_dir+'/DFTB+/dftb_in.hsd', 'w') as f:
            for i in range(len(lines)):
                f.write(lines[i])

        subprocess.run(current_dir+"/src/dftbOpt.sh", shell=True)

    def RunStaticOnFrame(self,idx,vdw=None): #runs a static calculation on the frame idx of the evolution
        geom = structures.Custom(self.evolution[idx])
        geom.types = self.types
        geom.SaveGeometry(charges=self.evolution[idx])
        geom.RunStatic(vdw,read_charges=True)

    def GetForcesOnFrame(self,idx,vdw=None): #returns to forces on the frame idx of the evolution
        geom = structures.Custom(self.evolution[idx])
        geom.types = self.types
        geom.SaveGeometry(charges=self.evolution[idx])
        geom.RunStatic(vdw,read_charges=True)
        return geom.GetForces()

    def GetDistOnFrame(self,idx):
        geom = structures.Custom(self.evolution[idx])
        geom.UpdateR0s()
        geom.SaveDistances()

    def SaveEvolutionAs(self, name): #saves the MD xyz simulation with a different name
        shutil.copyfile(current_dir+'/DFTB+/geo_end.xyz', current_dir+'/DFTB+/'+name+'.xyz')

    def SaveElectronChargesOnFrame(self,idx):
        geom = structures.Custom(self.evolution[idx])
        geom.SaveGeometry(charges=self.evolution[idx])
        self.structure_eq.SaveGeometry()

    def RetStructFromEvol(self,idx):
        geom = structures.Custom(self.evolution[idx])
        geom.types = self.structure_eq.types

        return geom

    def DecomposeTrajectory(self,name,path):
        for i in range(len(self.evolution)):
            print("Decomposition "+str(i)+"/"+str(len(self.evolution))+"    "+name)
            geom = structures.Custom(self.evolution[i])
            geom.SaveGeometry(decour=name+"_"+str(i),path=path,instruct=True,decour_charges=name+"_"+str(i),charges=self.evolution[i])
            self.structure_eq.SaveGeometry()

    def SaveLastFrame(self,name):
        print("Decomposition "+str(len(self.evolution))+"    "+name)
        geom = structures.Custom(self.evolution[len(self.evolution)-1])
        geom.SaveGeometry(decour=name,path=current_dir+"/out/")
        self.structure_eq.SaveGeometry()

    def LoadEvolution(self,path=None):
        angstrom = 0.529177249
        femtosecond = 41.341374575751
        if path == None:
            Nat, Niter, self.evolution, self.types = filetypes.Loadxyz(current_dir+"/DFTB+/geo_end.xyz",angstrom)
        else:
            Nat, Niter, self.evolution, self.types = filetypes.Loadxyz(path,angstrom)
        self.acelerations = []
        for i in range(len(self.evolution)-1):
            self.forcespmassiter = []
            for j in range(len(self.evolution[0])):
                self.forcespmassiter.append([(self.evolution[i+1][j][3]-self.evolution[i][j][3])/femtosecond,(self.evolution[i+1][j][4]-self.evolution[i][j][4])/femtosecond,(self.evolution[i+1][j][5]-self.evolution[i][j][5])/femtosecond])
            self.acelerations.append(self.forcespmassiter)

    def ComputeKenergy(self,min_steps=0,max_steps=None):
        if self.evolution == None:
            print ("Evolution not loaded")
            sys.exit()
        if len(self.evolution[0]) != self.structure_eq.Nat:
            print ("Evolution and equilibrium structure are not the same!")
            sys.exit()
        print("Computing kinetic energy..")
        Ken = []
        if max_steps == None:
            max_steps = len(self.evolution)
        for i in range(min_steps,max_steps):
            totalK = 0
            for j in range(len(self.evolution[0])):
                totalK = totalK + (self.evolution[i][j][3]*self.evolution[i][j][3]+self.evolution[i][j][4]*self.evolution[i][j][4]+self.evolution[i][j][5]*self.evolution[i][j][5])
            Ken.append(totalK*0.5*21896.1476887)
        return Ken

    def ComputeVenergy(self,vdw=None,min_steps=0,max_steps=None):
        if self.evolution == None:
            print ("Evolution not loaded")
            sys.exit()
        if len(self.evolution[0]) != self.structure_eq.Nat:
            print ("Evolution and equilibrium structure are not the same!")
            sys.exit()
        print("Computing potential energy..")
        Ven = []
        if max_steps == None:
            max_steps = len(self.evolution)
        for i in range(min_steps,max_steps):
            print("iter "+str(i)+"/"+str(max_steps))
            self.RunStaticOnFrame(i,vdw)
            TotalV = self.structure_eq.GetEnergy()
            Ven.append(TotalV)
        self.structure_eq.SaveGeometry()
        return Ven

    def ComputeTempDispersions(self):
        if self.evolution == None:
            print ("Evolution not loaded")
            sys.exit()
        if len(self.evolution[0]) != self.structure_eq.Nat:
            print ("Evolution and equilibrium structure are not the same")
            sys.exit()

        averages = []
        for i in range(len(self.evolution[0])):
            averagex = 0
            averagey = 0
            averagez = 0
            for j in range(len(self.evolution)):
                averagex = averagex + self.evolution[j][i][0]
                averagey = averagey + self.evolution[j][i][1]
                averagez = averagez + self.evolution[j][i][2]
            averages.append([averagex/len(self.evolution),averagey/len(self.evolution),averagez/len(self.evolution)])
        dispersions = []
        for i in range(len(self.evolution[0])):
            dispersionx = 0
            dispersiony = 0
            dispersionz = 0
            for j in range(len(self.evolution)):
                dispersionx = dispersionx + (self.evolution[j][i][0]-averages[i][0])*(self.evolution[j][i][0]-averages[i][0])
                dispersiony = dispersiony + (self.evolution[j][i][1]-averages[i][1])*(self.evolution[j][i][1]-averages[i][1])
                dispersionz = dispersionz + (self.evolution[j][i][2]-averages[i][2])*(self.evolution[j][i][2]-averages[i][2])
            dispersions.append(np.sqrt((dispersionx+dispersiony+dispersionz))/len(self.evolution))

        averager = []
        for i in range(len(self.evolution[0])):
            r = 0
            for j in range(len(self.evolution)):
                diffx = (self.evolution[j][i][0]-averages[i][0])
                diffy = (self.evolution[j][i][1]-averages[i][1])
                diffz = (self.evolution[j][i][2]-averages[i][2])
                r = r + np.sqrt(diffx*diffx+diffy*diffy+diffz*diffz)
            averager.append(r/len(self.evolution))


        return averages, dispersions, averager

    def ComputeBondDispersions(self):
        if self.evolution == None:
            print ("Evolution not loaded")
            sys.exit()
        if len(self.evolution[0]) != self.structure_eq.Nat:
            print ("Evolution and equilibrium structure are not the same")
            sys.exit()

        bonds = self.structure_eq.bonds

        

        averages = []
        for i in range(len(self.evolution[0])):
            averages_in = []
            for j in range(len(bonds[i])):
                bx = 0
                by = 0
                bz = 0
                br = 0
                for k in range(len(self.evolution)):
                    bx = self.evolution[k][i][0]-self.evolution[k][bonds[i][j]][0]
                    by = self.evolution[k][i][1]-self.evolution[k][bonds[i][j]][1]
                    bz = self.evolution[k][i][2]-self.evolution[k][bonds[i][j]][2]
                    br = br + np.sqrt(bx*bx+by*by+bz*bz)
                averages_in.append(br/len(self.evolution))
            averages.append(averages_in)

        dispersions = []
        for i in range(len(self.evolution[0])):
            for j in range(len(bonds[i])):
                bx = 0
                by = 0
                bz = 0
                br = 0
                dispersion = 0
                for k in range(len(self.evolution)):
                    bx = self.evolution[k][i][0]-self.evolution[k][bonds[i][j]][0]
                    by = self.evolution[k][i][1]-self.evolution[k][bonds[i][j]][1]
                    bz = self.evolution[k][i][2]-self.evolution[k][bonds[i][j]][2]
                    br = np.sqrt(bx*bx+by*by+bz*bz)
                    dispersion = dispersion + (br-averages[i][j])*(br-averages[i][j])
                dispersions.append(np.sqrt(dispersion/len(self.evolution)))

        return averages, dispersions

    def GetForcesSOE(self,vdw=None):
        structure = copy.deepcopy(self.structure_eq)
        Forces=[]
        self.LoadEvolution()

        for i in range(len(self.evolution)):
            print(str(i)+"/"+str(len(self.evolution))+"\n")
            for j in range(structure.Nat):
                structure.x[j]=self.evolution[i][j][0]
                structure.y[j]=self.evolution[i][j][1]
                structure.z[j]=self.evolution[i][j][2]
            structure.SaveGeometry()
            structure.RunStatic(vdw)
            Forces.append(structure.GetForces())
        return Forces

