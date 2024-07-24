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

## Contributing

Contributions to the toolkit are welcome! Please read `CONTRIBUTING.md` for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.

## Acknowledgments

- Thanks to the DFTB+ community for providing the simulation software.
- Acknowledge any collaborators or institutions that supported the development of this toolkit.
