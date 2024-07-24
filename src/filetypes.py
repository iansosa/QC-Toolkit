import numpy as np
import numpy.linalg as LA
import sys

def Loadsdf(path,conversion):
    """
    Loads molecular geometry from an SDF file.

    Parameters:
    - path (str): Path to the SDF file.
    - conversion (float): Conversion factor from the units in the file to Bohr.

    Returns:
    - Nat (int): Number of atoms.
    - geometry (list): List of atomic positions.
    """
    try:
        file = open(path, "r+")
    except OSError:
        print ("Could not open sdf file")
        sys.exit()
    lines = file.readlines()
    aux = lines[3].split(' ')
    aux = list(filter(lambda x: x != '', aux))
    Nat = int(aux[0])
    lines = lines[4:]

    geometry = []
    noC = 0
    for i in range(Nat):
        a = lines[i].split(' ')
        a = list(filter(lambda x: x != '', a))
        if a[3] == 'C':
            a = list(map(float, a[0:3]))
            geometry.append(a)
        else:
            noC = noC + 1

    arr_t = np.array(geometry).T/conversion
    geometry = arr_t.tolist()
    Nat = Nat - noC
    return Nat, geometry

def Loadgro(path,conversion):
    """
    Loads molecular geometry from a GRO file.

    Parameters:
    - path (str): Path to the GRO file.
    - conversion (float): Conversion factor from the units in the file to Bohr.

    Returns:
    - Nat (int): Number of atoms.
    - geometry (list): List of atomic positions.
    - periodic (bool): Indicates if the system is periodic.
    - aux_types (list): List of atom types.
    - unit_cell (list): List of unit cell vectors.
    """
    try:
        file = open(path, "r+")
    except OSError:
        print ("Could not open gro file "+ path)
        sys.exit()
    lines = file.readlines()

    aux = lines[1]
    Nat = int(aux)
    print(Nat)

    lines = lines[2:]

    aux_types = []

    geometry = []
    for i in range(Nat):
        aux_types.append(lines[i][13])
        geometry.append([float(lines[i][21:28]),float(lines[i][29:36]),float(lines[i][37:44])])
    arr_t = np.array(geometry).T/conversion
    geometry = arr_t.tolist()


    unit_cell = []
    a = lines[Nat].split(' ')
    a = list(filter(lambda x: x != '', a))
    a = list(map(float, a))
    unit_cell = [[0,0,0],[a[0],0,0],[0,a[1],0],[0,0,a[2]]]
    arr_t = np.array(unit_cell)/conversion
    unit_cell = arr_t.tolist()

    periodic = True
    return Nat, geometry, periodic, aux_types, unit_cell

def Loadgen(path,conversion):
    """
    Loads molecular geometry from a GEN file.

    Parameters:
    - path (str): Path to the GEN file.
    - conversion (float): Conversion factor from the units in the file to Bohr.

    Returns:
    - Nat (int): Number of atoms.
    - geometry (list): List of atomic positions.
    - periodic (bool): Indicates if the system is periodic.
    - aux_types (list): List of atom types.
    - unit_cell (list): List of unit cell vectors.
    """
    try:
        file = open(path, "r+")
    except OSError:
        print ("Could not open gen file "+ path)
        sys.exit()
    lines = file.readlines()

    aux = lines[0].split(' ')
    aux = list(filter(lambda x: x != '', aux))
    Nat = int(aux[0])

    types = lines[1].split(' ')
    types = list(filter(lambda x: x != '', types))
    types[-1] = types[-1].strip("\n")

    periodic = False
    if len(aux) > 0:
        if aux[1] == 'S\n':
            periodic = True

    lines = lines[2:]

    aux_types = []

    geometry = []
    for i in range(Nat):
        a = lines[i].split(' ')
        a = list(filter(lambda x: x != '', a))
        aux_types.append(a[1])
        a = list(map(float, a[2:]))
        geometry.append(a)
    arr_t = np.array(geometry).T/conversion
    geometry = arr_t.tolist()

    lines = lines[Nat:]
    for i in range(len(aux_types)):
        if aux_types[i] == '1':
            aux_types[i] = types[0]
        if aux_types[i] == '2':
            aux_types[i] = types[1]
        if aux_types[i] == '3':
            aux_types[i] = types[2]

    unit_cell = []
    for i in range(len(lines)):
        a = lines[i].split(' ')
        a = list(filter(lambda x: x != '', a))
        a = list(map(float, a))
        unit_cell.append(a)
    arr_t = np.array(unit_cell)/conversion
    unit_cell = arr_t.tolist()

    # print(aux_types)
    return Nat, geometry, periodic, aux_types, unit_cell

def Loadcc1(path,conversion):
    """
    Loads molecular geometry from a CC1 file.

    Parameters:
    - path (str): Path to the CC1 file.
    - conversion (float): Conversion factor from the units in the file to Bohr.

    Returns:
    - Nat (int): Number of atoms.
    - geometry (list): List of atomic positions.
    - aux_types (list): List of atom types.
    """
    try:
        file = open(path, "r+")
    except OSError:
        print ("Could not open cc1 file")
        sys.exit()
    lines = file.readlines()

    aux = lines[0].split(' ')
    aux = list(filter(lambda x: x != '', aux))
    Nat = int(aux[0])
    lines = lines[1:]

    geometry = []
    for i in range(len(lines)):
        a = lines[i].split(' ')
        a = list(filter(lambda x: x != '', a))
        a = list(map(float, a[2:5]))
        geometry.append(a)

    arr_t = np.array(geometry).T/conversion
    geometry = arr_t.tolist()

    aux_types = []
    for i in range(Nat):
        aux_types.append('C')
    return Nat, geometry, aux_types

def Loadtxt(path,conversion):
    """
    Loads molecular geometry from a TXT file.

    Parameters:
    - path (str): Path to the TXT file.
    - conversion (float): Conversion factor from the units in the file to Bohr.

    Returns:
    - Nat (int): Number of atoms.
    - geometry (list): List of atomic positions.
    """
    try:
        file = open(path, "r+")
    except OSError:
        print ("Could not open cc1 file")
        sys.exit()
    lines = file.readlines()

    Nat = len(lines)

    geometry = []
    for i in range(len(lines)):
        a = lines[i].split(' ')
        a = list(filter(lambda x: x != '', a))
        a = list(map(float, a))
        geometry.append(a)

    arr_t = np.array(geometry).T/conversion
    geometry = arr_t.tolist()

    return Nat, geometry

def Loadxyz_single(path,conversion):
    """
    Loads a single frame of molecular geometry from an XYZ file.

    Parameters:
    - path (str): Path to the XYZ file.
    - conversion (float): Conversion factor from the units in the file to Bohr.

    Returns:
    - Nat (int): Number of atoms.
    - geometry (list): List of atomic positions.
    """
    try:
        file = open(path, "r+")
    except OSError:
        print ("Could not open xyz file")
        sys.exit()
    lines = file.readlines()

    aux = lines[0].split(' ')
    aux = list(filter(lambda x: x != '', aux))
    Nat = int(aux[0])
    lines = lines[2:]

    geometry = []
    for i in range(len(lines)):
        a = lines[i].split(' ')
        a = list(filter(lambda x: x != '', a))
        a = list(map(float, a[1:-1]))
        geometry.append(a)
    arr_t = np.array(geometry).T/conversion
    geometry = arr_t.tolist()

    return Nat, geometry

def Loadxyz(path,conversion): #dftb+ specifies velocities in A/ps, velocities are returned in bohr/a.u(time)
    """
    Loads molecular dynamics trajectory from an XYZ file.

    Parameters:
    - path (str): Path to the XYZ file.
    - conversion (float): Conversion factor from the units in the file to Bohr.

    Returns:
    - Nat (int): Number of atoms.
    - Niter (int): Number of iterations (frames) in the trajectory.
    - evolution (list): List of atomic positions for each frame.
    - types (list): List of atom types.
    """
    print("Loadingxyz..")
    femtosecond = 41.341374575751
    try:
        file = open(path, "r+")
    except OSError:
        print ("Could not open xyz file "+path)
        sys.exit()
    lines = file.readlines()

    aux = lines[0].split(' ')
    aux = list(filter(lambda x: x != '', aux))
    Nat = int(aux[0])
    Niter = int(len(lines)/(2+Nat))

    evolution = []
    for i in range(Niter):
        instant = []
        types = []
        for j in range(Nat):
            a = lines[2+j+i*(2+Nat)].split(' ')
            a = list(filter(lambda x: x != '', a))
            types.append(a[0])
            a = list(map(float, a[1:8]))
            c = a[3]
            a = [a[0]/conversion,a[1]/conversion,a[2]/conversion,a[4]/(femtosecond*conversion*1000),a[5]/(femtosecond*conversion*1000),a[6]/(femtosecond*conversion*1000)]
            a.append(c)
            instant.append(a)
        evolution.append(instant)
        
    return Nat, Niter, evolution, types

def LoadPly(path,Radius):
    """
    Loads molecular geometry from a PLY file.

    Parameters:
    - path (str): Path to the PLY file.
    - Radius (float): Desired radius to scale the geometry.

    Returns:
    - Nat (int): Number of atoms.
    - geometry (list): List of atomic positions.
    """
    print("Loadingply...")
    try:
        file = open(path, "r+")
    except OSError:
        print ("Could not open ply file "+path)
        sys.exit()
    lines = file.readlines()
    lines = lines[8:]

    geometry = []
    for i in range(len(lines)):
        a = lines[i].split(' ')
        a = list(filter(lambda x: x != '', a))
        a = list(map(float,a))
        geometry.append(np.array(a))
    geometry = list(np.unique(np.array(geometry),axis=0))
    Nat = len(geometry)
    avg = 0
    for i in range(len(geometry)):
        avg = avg + np.array(geometry[i])
    avg = avg / len(geometry)
    geometry = geometry - avg

    avg = 0
    for i in range(len(geometry)):
        avg = avg + LA.norm(geometry[i])
    avg = avg / len(geometry)
    arr_t = np.array(geometry).T*Radius/avg

    geometry = arr_t.tolist()

    return Nat, geometry