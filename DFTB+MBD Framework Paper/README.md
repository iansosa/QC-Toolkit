# Examples for the DFTB+MBD Framework Paper

This folder was buit to support the following paper published in IJES:
```
@article{DFTB+MBD,
title = {Quantum-informed simulations for mechanics of materials: DFTB+MBD framework},
journal = {International Journal of Engineering Science},
author={Zhaoxiang Shen and Raúl I. Sosa and Stéphane P. A. Bordas and Alexandre Tkatchenko and Jakub Lengiewicz},
volume = {204},
pages = {104126},
year = {2024},
issn = {0020-7225},
doi = {https://doi.org/10.1016/j.ijengsci.2024.104126}}

```
The folder includes examples of three different molecular systems, namely carbon chains, carbon nanotubes, and Ultra High Molecular Weight Polyethylene (UHMWPE), which correponds to Section 4 in the paper.

# TensorFlow implementations
Some examples were conducted by TensorFlow-based (TF) implementations, e.g. harmonic model for carbon-nanotube buckling. The vdW dispersion in the TF codes is a modified version of the TF implementation of [libmbd](https://github.com/libmbd/libmbd). We extend it to enable hessian calculation, and include additional variants of MBD (ts, scs, rsscs) and PW (TS, LJ) models. To run TF-based demos, [Tensorflow with GPU support](https://www.tensorflow.org/install) is required. 

