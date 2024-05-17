#!/bin/bash
. ~/miniconda3/etc/profile.d/conda.sh && conda activate dftb
# . /opt/apps/resif/iris/2020b/broadwell/software/Anaconda3/2020.11/etc/profile.d/conda.sh && conda activate dftb
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/../DFTB+" || {
    echo "Failed to change directory to DFTB+."
    exit 1
}
export OMP_NUM_THREADS=4
# cd DFTB+ && ./dftbplus_MBD-20200428_mkl2018-mpiscalapack.x
dftb+