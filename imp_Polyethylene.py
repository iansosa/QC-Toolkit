import numpy as np
import numpy.linalg as LA
from mdhandler import Handler as MDH
from bondcalc import Bonds
import structures
import sys
from tqdm import tqdm
import random
from heapq import nsmallest
from operator import itemgetter
import statistics

def vol_convergence_test():
    Nneighbours = 150

    geom = structures.FromFile("Polyethylene/Greek/confin100000.gro")
    geom.Fold()
    geom.Expand([2,2,2],trim = 1.2)
    idx_range = list(range(0, 300050))
    random.shuffle(idx_range)

    for i in range(0,1000):
        a = geom.Distances(idx_range[0])
        idx, _ = zip(*nsmallest(Nneighbours, enumerate(a), key=itemgetter(1)))
        geom_aux = structures.Custom(geom.PosAsListIdx(idx),types=geom.TypesAsListIdx(idx))
        geom_aux.SaveGeometry()
        geom_aux.RunStatic(get_CPA=True)

        volume = geom_aux.GetVolume(0)

        with open("out/volumes_conv_test_"+str(Nneighbours)+".txt", 'a') as f:
            f.write(str(volume)+"\n")


def infer_vol_structure(Nneighbours = 150,from_idx=0, to_idx=300050,load_vol=True):
    

    geom = structures.FromFile("Polyethylene/Greek/confin100000.gro")
    geom.Fold()
    geom.Expand([2,2,2],trim = 1.2)

    idx_range = list(range(from_idx, to_idx))

    if load_vol == True:
        try:
            file = open("out/volumes_"+str(Nneighbours)+"_"+str(from_idx)+".txt", "r+")
        except OSError:
            print ("File does not exist: "+"out/volumes_"+str(Nneighbours)+"_"+str(from_idx)+".txt")
            sys.exit()
        lines = file.readlines()
        content = []
        for i in range(len(lines)):
            a = lines[i].split(' ')
            a = list(filter(lambda x: x != '', a))
            a = list(map(float, a))
            content.append(a)
        content = np.array(content).T
        idx_range = list(range(int(content[0][len(lines)-1]+1), to_idx))

    for i in idx_range:
        print("inference: "+str(i)+" ;from: "+str(from_idx)+" ;to: "+str(to_idx)+" ;percentage: "+str(100*(i-from_idx)/(to_idx-from_idx))+"\n")
        a = geom.Distances(i)
        idx, _ = zip(*nsmallest(Nneighbours, enumerate(a), key=itemgetter(1)))
        geom_aux = structures.Custom(geom.PosAsListIdx(idx),types=geom.TypesAsListIdx(idx))
        geom_aux.SaveGeometry()
        geom_aux.RunStatic(get_CPA=True)

        volume = geom_aux.GetVolume(0)

        with open("out/volumes_"+str(Nneighbours)+"_"+str(from_idx)+".txt", 'a') as f:
            f.write(str(i)+" "+str(volume)+"\n")


def melt_structure_natoms(natoms,start_idx,end_idx):
    N=20
    R0=2.5
    geom = structures.Sphere(N,R0)
    geom.LoadGeometry("Polyethylene/Greek/confin100000.gro")
    geom.Fold()
    # geom.SaveGeometry()
    geom.Expand([2,2,2],trim = 1.2)


    idx_range = list(range(start_idx, end_idx))
    random.shuffle(idx_range)

    for i in idx_range:
        a = geom.Distances(i)
        dist = nsmallest(natoms, a)
        print(dist)
        idx, _ = zip(*nsmallest(natoms, enumerate(a), key=itemgetter(1)))
        geom_aux = structures.Custom(geom.PosAsListIdx(idx),types=geom.TypesAsListIdx(idx))
        geom_aux.SaveGeometry()
        geom_aux.RunStatic()
        with open("out/"+str(natoms)+"_atom_radius.txt", 'a') as f:
            f.write(str(geom.Distance(i,idx[natoms-1]))+" \n")


def get_volume(Nneighbours):
    N=20
    R0=2.5
    geom = structures.Sphere(N,R0)
    geom.LoadGeometry("Polyethylene/Greek/confin100000.gro")
    geom.Fold()
    geom.Expand([2,2,2],trim = 1.2)
    idx_range = list(range(0, 300050))
    random.shuffle(idx_range)

    C_vol= []
    H_vol= []
    for i in idx_range[:3000]:
        a = geom.Distances(i)
        idx, _ = zip(*nsmallest(Nneighbours, enumerate(a), key=itemgetter(1)))
        geom_aux = structures.Custom(geom.PosAsListIdx(idx),types=geom.TypesAsListIdx(idx))
        geom_aux.SaveGeometry()
        geom_aux.RunStatic(get_CPA=True)

        try:
            file = open("DFTB+/CPA_ratios.out", "r+")
        except OSError:
            print ("Could not CPA_ratios file "+ path)
            sys.exit()
        lines = file.readlines()

        volume = lines[1].split(' ')
        volume = list(filter(lambda x: x != '', volume))
        volume = float(volume[1])
        if geom.types[i] == 'C':
            C_vol.append(volume)
        if geom.types[i] == 'H':
            H_vol.append(volume)
    avg_C = np.mean(C_vol)
    std_C = np.std(C_vol,ddof=1)
    avg_H = np.mean(H_vol)
    std_H = np.std(H_vol,ddof=1)
    with open("out/volumes_"+str(Nneighbours)+".txt", 'w') as f:
        f.write("C "+str(avg_C)+" "+str(std_C)+"\n")
        f.write("H "+str(avg_H)+" "+str(std_H)+"\n")

def get_volumeconvergence():
    N=20
    R0=2.5
    geom = structures.Sphere(N,R0)
    geom.LoadGeometry("Polyethylene/Greek/confin100000.gro")
    geom.Fold()
    geom.Expand([2,2,2],trim = 1.2)
    idx_range = list(range(0, 300050))
    random.shuffle(idx_range)

    vec_C = []
    vec_H = []

    n_stats = 10
    n_size = 15
    n_c = 0
    n_h = 0
    for i in idx_range:
        C_vol= []
        H_vol= []
        if n_c == n_stats and n_h == n_stats:
            break
        if n_c == n_stats and geom.types[i] == 'C':
            continue
        if n_h == n_stats and geom.types[i] == 'H':
            continue
        if geom.types[i] == 'C':
            n_c = n_c + 1
        if geom.types[i] == 'H':
            n_h = n_h + 1

        a = geom.Distances(i)
        for j in range(n_size):
            idx, _ = zip(*nsmallest(10+j*20, enumerate(a), key=itemgetter(1)))
            geom_aux = structures.Custom(geom.PosAsListIdx(idx),types=geom.TypesAsListIdx(idx))
            geom_aux.SaveGeometry()
            geom_aux.RunStatic(get_CPA=True)

            try:
                file = open("DFTB+/CPA_ratios.out", "r+")
            except OSError:
                print ("Could not CPA_ratios file "+ path)
                sys.exit()
            lines = file.readlines()

            volume = lines[1].split(' ')
            volume = list(filter(lambda x: x != '', volume))
            volume = float(volume[1])
            if geom.types[i] == 'C':
                C_vol.append(volume)
                print(C_vol)
            if geom.types[i] == 'H':
                H_vol.append(volume)
                print(H_vol)
        if geom.types[i] == 'C':
            vec_C.append(C_vol)
        if geom.types[i] == 'H':
            vec_H.append(H_vol)
    with open("out/volumes_conv_C.txt", 'w') as f:
        for i in range(len(vec_C)):
            for j in range(len(vec_C[i])):
                f.write(str(vec_C[i][j]/vec_C[i][len(vec_C[i])-1])+" ")
            f.write("\n")
    with open("out/volumes_conv_H.txt", 'w') as f:
        for i in range(len(vec_H)):
            for j in range(len(vec_H[i])):
                f.write(str(vec_H[i][j]/vec_H[i][len(vec_H[i])-1])+" ")
            f.write("\n")


def md_analyze_corr(evol_filepath):
    N=20
    R0=2.5
    geom = structures.Sphere(N,R0)
    geom.LoadGeometry("Polyethylene/PE_chains_shorterer_capped.gen")



    ############################################################################ get triplets; can be done once
    components = geom.ConectedComponents()
    assert len(components) == 2, "number of compnents different than 2"
    component1 = components[0]
    component2 = components[1]

    triplet1 = []
    triplet2 = []

    for i in range(len(geom.bonds)):
        if len(geom.bonds[i]) == 4:
            if i in component1:
                hid = []
                for j in range(4):
                    if geom.types[geom.bonds[i][j]] == "H":
                        hid.append(geom.bonds[i][j])
                triplet1.append([i,hid[0],hid[1]])
            if i in component2:
                hid = []
                for j in range(4):
                    if geom.types[geom.bonds[i][j]] == "H":
                        hid.append(geom.bonds[i][j])
                triplet2.append([i,hid[0],hid[1]])
    ####################################################################################################### get versors; should be done every time

    hist_inner = []
    hist_sigma_inner = []
    hist_sigma_dist = []

    md = MDH(geom,False)
    md.LoadEvolution(evol_filepath)
    for j  in tqdm(range(len(md.evolution)),desc="Progress"):
        geom = md.RetStructFromEvol(j)
        versors1 = []
        for i in range(len(triplet1)):
            versors1.append([triplet1[i][0],(geom.GetVersor(triplet1[i][0],triplet1[i][1])+geom.GetVersor(triplet1[i][0],triplet1[i][2]))/LA.norm(geom.GetVersor(triplet1[i][0],triplet1[i][1])+geom.GetVersor(triplet1[i][0],triplet1[i][2]))])
        versors2 = []
        for i in range(len(triplet2)):
            versors2.append([triplet2[i][0],(geom.GetVersor(triplet2[i][0],triplet2[i][1])+geom.GetVersor(triplet2[i][0],triplet2[i][2]))/LA.norm(geom.GetVersor(triplet2[i][0],triplet2[i][1])+geom.GetVersor(triplet2[i][0],triplet2[i][2]))])


        avg_inner = 0
            
        avg_min_dist = 0
        for i in range(len(versors1)):
            inner_aux = 0
            min_dist = 10000
            for j in range(len(versors2)):
                if min_dist > geom.Distance(versors1[i][0],versors2[j][0]):
                    min_dist = geom.Distance(versors1[i][0],versors2[j][0])
                    inner_aux = np.clip(np.dot(versors1[i][1],versors2[j][1]),-1.0,1.0)

            hist_inner.append(np.abs(inner_aux))

            avg_min_dist = avg_min_dist + min_dist
            avg_inner = avg_inner + np.abs(inner_aux)
        avg_min_dist = avg_min_dist/len(versors1)
        avg_inner = avg_inner/len(versors1)
        min_dist = 10000
        sigma_dist = 0
        sigma_inner = 0
        for i in range(len(versors1)):
            for j in range(len(versors2)):
                if min_dist > geom.Distance(versors1[i][0],versors2[j][0]):
                    min_dist = geom.Distance(versors1[i][0],versors2[j][0])
                    inner_aux = np.clip(np.dot(versors1[i][1],versors2[j][1]),-1.0,1.0)
            sigma_dist = sigma_dist + (avg_min_dist - min_dist)**2
            sigma_inner = sigma_inner + (avg_inner-np.abs(inner_aux)) **2

        sigma_dist = sigma_dist/len(versors1)
        hist_sigma_dist.append(sigma_dist)
        sigma_inner = sigma_inner/len(versors1)
        hist_sigma_inner.append(sigma_inner)

    utils._write_file("histogram-microangle.txt",hist_inner)
    utils._write_file("histogram-sigma-angle.txt",hist_sigma_inner)
    utils._write_file("histogram-sigma-dist.txt",hist_sigma_dist)

def long_md(name,T,vdw=None):
    struct_name = name+".gen"

    geom = structures.FromFile(struct_name)
    Nat = geom.Nat
    geom2 = structures.FromFile(struct_name)
    geom2.MoveAll([0,10,0])
    geom.add(geom2)
    geom.SaveGeometry()
    md = MDH(geom,False)
    md.RunMD(steps=1500000,temp=T,vdw=vdw,keepstationary=False,static=[66,48,Nat+66,Nat+48])