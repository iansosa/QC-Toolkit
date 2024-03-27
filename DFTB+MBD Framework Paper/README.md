# Examples for the DFTB+MBD Framework Paper

This folder was buit to support the following paper
```
@article{DFTB+MBD,
title = {Quantum-informed simulations for mechanics of materials: DFTB+MBD framework},
journal = {TBD},
author = {Z. Shen and R.I. Sosa and S.P.A. Bordas and A. Tktchenko and J. Lengiewicz.},
}
```
The folder includes examples of three different molecular systems, namely carbon chains, carbon nanotubes, and Ultra High Molecular Weight Polyethylene (UHMWPE), which correponds to Section 4 in the paper.

# TensorFlow implementations
Some examples were conducted by TensorFlow-based (TF) implementations, e.g. harmonic model for carbon-nanotube buckling. The vdW dispersion in the TF codes is a modified version of the TF implementation of [libmbd](https://github.com/libmbd/libmbd). We extend it to enable hessian calculation, and include additional variants of MBD (ts, scs, rsscs) and PW (TS, LJ) models. To run TF-based demos, [Tensorflow with GPU support](https://www.tensorflow.org/install) is required. 

