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

The toolkit consists of several Python scripts and shell scripts located in the `src/` directory. Here is a brief overview of how to use these scripts:

### Geometry Manipulation

To manipulate molecular geometries, use the `structures.py` script:

```python
from structures import Sphere, Ring, Chain

# Create a new spherical structure with 30 atoms and an interatomic distance of 2.4 Bohr
sphere = Sphere(30, 2.4)
```

### Simulation Setup

To set up a simulation, use the `dftbOpt.sh` shell script:

```bash
./src/dftbOpt.sh
```

This script will prepare the input files and execute DFTB+ with the appropriate settings.

### Data Analysis

Analyze simulation data using the `mdhandler.py` script:

```python
from mdhandler import Handler as MDH

# Load an MD trajectory and compute kinetic energy
md_handler = MDH(structure_eq, optimize=False)
md_handler.LoadEvolution('path/to/trajectory.xyz')
kinetic_energy = md_handler.ComputeKenergy()
```

### Parametrization

Adjust bond parameters using the `bondcalc.py` script:

```python
from bondcalc import Bonds

# Initialize bond calculations for a given structure
bonds = Bonds(structure)
```

## Contributing

Contributions to the toolkit are welcome! Please read `CONTRIBUTING.md` for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.

## Acknowledgments

- Thanks to the DFTB+ community for providing the simulation software.
- Acknowledge any collaborators or institutions that supported the development of this toolkit.
