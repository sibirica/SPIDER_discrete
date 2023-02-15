# ScalableSPCA.jl

Software supplement for the paper

"Solving Large-Scale Sparse PCA to Certifiable (Near) Optimality"

by Dimitris Bertsimas, Ryan Cory-Wright and Jean Pauphilet for which a preprint is available [here](https://arxiv.org/abs/2005.05195).

## Introduction

The software in this package is designed to provide certifiably near-optimal solutions to the problem

`max x'Qx`
`s.t. ||x||_0 <=k, ||x||_2 <=1`

using a relax-and-round approach.  The code implements various relax-and-round procedures described in the paper "Solving Large-Scale Sparse PCA to Certifiable (Near) Optimality"  by Dimitris Bertsimas, Ryan Cory-Wright and Jean Pauphilet.


## Installation and set up

In order to run this software, you must install a recent version of Julia from http://julialang.org/downloads/, and a recent version of the Mosek solver (academic licenses are freely available at https://www.mosek.com/products/academic-licenses/). If you do not have access to Mosek, you could use the freely available SCS package instead, although your results may vary.  The most recent version of Julia at the time this code was last tested was Julia 1.3.0 using Mosek version 9.0.

Several packages must be installed in Julia before the code can be run.  These packages can be found in "core_julia1p3.jl"

At this point, the "createTablex.jl" files should run successfully.  To run the script, navigate to the folder directory and run:

`include("createTablex.jl")`

 The script will reproduce table x in the paper, where x is an integer between 2 and 4 (we omitted the micromass dataset and table 5 due to the size of the datasets involved, but these results can readily be reproduced by downloading the Gisette/Arcene datasets from the UCI repository).  

## Use of the getSDPUpperBound_gd() function

The key method in this packages is getSDPUpperBound_gd().  It takes two required  arguments: `sigma`, and `k`, as well as two optional arguments which tell the method whether or not include the PSD constraint, and whether or not to include some additional valid inequalities which strengthen the relaxation. The variable `sigma` is a covariance matrix that holds the original data. The parameter `k` is a positive integer less than n.


## Related Packages
If you are interested in computing multiple provably near optimal PCs, you may want to check out our related package https://github.com/ryancorywright/MultipleComponentsSoftware


## Citing ScalableSPCA.jl

If you use SparsePortfolioSelection.jl, we ask that you please cite the following [paper](https://arxiv.org/abs/2005.05195):
```
@article{bertsimas2020solving,
  title={Solving Large-Scale Sparse PCA to Certifiable (Near) Optimality},
  author={Bertsimas, Dimitris and Cory-Wright, Ryan and Pauphilet, Jean},
  journal={arXiv preprint arXiv:2005.05195},
  year={2020}
}
```

## Thank you

Thank you for your interest in ScalableSPCA. Please let us know if you encounter any issues using this code, or have comments or questions.  Feel free to email us anytime.


Dimitris Bertsimas
dbertsim@mit.edu

Ryan Cory-Wright
ryancw@mit.edu

Jean Pauphilet
jpauphilet@london.edu
