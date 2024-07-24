# DFTB+ Toolkit/API

## Overview

This toolkit/API provides a suite of tools for working with the Density Functional Tight Binding Plus (DFTB+) software. It is designed to facilitate various tasks including geometry manipulation, simulation setup, data analysis on quasi-static simulations and dynamics, and parametrization of covalent bonds.

## Features

- Geometry manipulation: Easily adjust and optimize molecular structures.
- Simulation setup: Prepare input files and configurations for DFTB+ simulations.
- Data analysis: Analyze the results from quasi-static simulations and dynamics.
- Parametrization: Fine-tune covalent bond parameters for accurate simulations.

## Installation

To install the toolkit, clone the repository to your local machine using:

```bash
git clone https://github.com/iansosa/QC-Toolkit.git
cd QC-Toolkit
```

Ensure that you have the necessary dependencies installed, including DFTB+, Python, and any required Python packages.

## Usage

The toolkit is used by writing Python scripts that import and utilize the provided classes and functions. Below are some examples of how to use the toolkit.

### Geometry Manipulation

To manipulate molecular geometries, import and use the geometry classes:

```python
from src.structures import Sphere, Ring, Chain

# Create a new spherical structure with 30 atoms and an interatomic distance of 2.4 Bohr
sphere = Sphere(30, 2.4)
sphere.SetPos(30, 2.4)  # Set positions of atoms in the sphere
```

### Simulation Setup and Execution

Prepare and run simulations by creating a structure and using the `Handler` class to manage simulations:

```python
from src.geohandler import Handler
from src.structures import Custom

# Load a custom molecular structure from a file
custom_structure = Custom('path/to/geometry.xyz')

# Initialize the handler with the custom structure
handler = Handler(custom_structure)

# Optimize the geometry
handler.RunOptimize()

# Run a static calculation
handler.RunStatic()
```

### Data Analysis

Analyze simulation data using the methods provided by the `Handler` class:

```python
# Assuming 'handler' is already initialized with a structure

# Save the optimized geometry to a file
handler.SaveGeometry('optimized_geometry.gen')

# Load and analyze the forces on the structure
forces = handler.GetForces()
```

### Parametrization

Adjust bond parameters using the `Bonds` class:

```python
from src.bondcalc import Bonds

# Initialize bond calculations for the custom structure
bonds = Bonds(custom_structure)

# Calculate and save the bond matrix
bonds.CalcSaveBondMatrix()
```

## Example: Buckling Test on a Carbon Nanotube

This example demonstrates how to perform a buckling test on a carbon nanotube using the toolkit. The test involves compressing the nanotube and analyzing the forces experienced by the atoms at the boundaries.

### Buckling Simulation

The `buckling_simulation` function compresses the nanotube by a specified displacement increment and performs an optimization at each step. The geometry is saved after each optimization.

```python
from structures import FromFile

def buckling_simulation(vdw=None):
    # Select the appropriate structure file based on the van der Waals (vdW) method
    struct_name = f"Nanotubes/buckling/{'novdw' if vdw is None else vdw}/original.gen"

    # Load the nanotube structure from the file
    Nanotube = FromFile(struct_name)
    Nat = Nanotube.Nat
    print(Nat)

    # Define the boundary atoms of the nanotube
    Nanotube_boundaries = [...]
    Nanotube_left_boundary = [...]

    # Save the initial geometry
    Nanotube.SaveGeometry()

    # Perform the buckling simulation
    dcomp = 5.0 / 100.0
    for i in range(0, 100):
        total_comp = i * dcomp
        ru = str(round(total_comp, 2)).ljust(4, '0')

        # Optimize the structure with the current compression
        Nanotube.RunOptimize(vdw=vdw, static=Nanotube_boundaries, read_charges=False)
        Nanotube.LoadGeometry()

        # Save the optimized geometry
        filename = f"geom_{'novdw' if vdw is None else vdw}_{ru}"
        Nanotube.SaveGeometry(filename, "buckling")

        # Apply additional compression for the next iteration
        for j in Nanotube_left_boundary:
            Nanotube.Displace(j, [0, 0, dcomp])
```

### Buckling Statistics

The `buckling_statistics` function calculates the total force experienced by the boundary atoms after each compression step and records the results.

```python
def buckling_statistics(vdw=None):
    # Define the boundary atoms and the axis to analyze
    Nanotube_left_boundary = [...]
    axis = 2  # z-axis

    # Perform the statistics calculation
    dcomp = 5.0 / 100.0
    for i in range(0, 100):
        total_comp = i * dcomp
        ru = str(round(total_comp, 2)).ljust(4, '0')

        # Load the structure for the current compression step
        filename = f"Nanotubes/buckling/{'novdw' if vdw is None else vdw}/geom_{'novdw' if vdw is None else vdw}_{ru}.gen"
        Nanotube = FromFile(filename)

        # Run a static calculation and get the forces
        Nanotube.RunStatic(vdw=vdw, read_charges=False)
        F = Nanotube.GetForces()

        # Calculate the total force on the boundary atoms
        F_total = sum(F[j][axis] for j in Nanotube_left_boundary)

        # Record the compression and total force
        with open('../out/Buckling.txt', 'a') as f:
            f.write(f"{ru} {F_total}\n")

# Uncomment the function you want to run
# buckling_simulation()
# buckling_statistics()
```

To run the buckling test, uncomment the desired function call at the bottom of the script. Ensure that the `Nanotubes/buckling/` directory contains the initial geometry files for the nanotube with the appropriate naming convention.

## Running the Example

To execute the buckling test example, navigate to the directory containing `Buckling_test_nanotube.py` and run the script with Python:

```bash
python Buckling_test_nanotube.py
```

The script will perform the buckling simulation or statistics calculation based on the function you choose to run. The results will be saved in the `out/` directory.