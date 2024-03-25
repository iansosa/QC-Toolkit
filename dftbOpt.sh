# . ~/miniconda3/etc/profile.d/conda.sh && conda activate dftb
. /opt/apps/resif/iris/2020b/broadwell/software/Anaconda3/2020.11/etc/profile.d/conda.sh && conda activate dftb
export OMP_NUM_THREADS=4
# cd DFTB+ && ./dftbplus_MBD-20200428_mkl2018-mpiscalapack.x
cd DFTB+ && dftb+