#!/bin/bash
# This script sets up the environment for running DFTB+ and executes the DFTB+ program.

# Load the conda environment for DFTB+. Replace 'miniconda3' and 'dftb' with the appropriate paths and environment names for your setup.
. ~/miniconda3/etc/profile.d/conda.sh && conda activate dftb
# Alternatively, uncomment the following line and comment out the previous line if using a different setup, such as a module on a cluster.
# . /opt/apps/resif/iris/2020b/broadwell/software/Anaconda3/2020.11/etc/profile.d/conda.sh && conda activate dftb

# Store the directory of the script in a variable.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Change the working directory to the DFTB+ directory. Exit with an error message if the directory change fails.
cd "${SCRIPT_DIR}/../DFTB+" || {
    echo "Failed to change directory to DFTB+."
    exit 1
}

# Set the number of threads for OpenMP parallelization.
export OMP_NUM_THREADS=4

# Run the DFTB+ executable. Replace 'dftb+' with the appropriate executable for your setup.
dftb+