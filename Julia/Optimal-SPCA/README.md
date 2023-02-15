# Optimal-SPCA

Software supplement for the paper

"Certifiably Optimal Principal Component Analysis"

by Lauren Berk and Dimitris Bertsimas

[![DOI](https://zenodo.org/badge/160833968.svg)](https://zenodo.org/badge/latestdoi/160833968)


## Introduction

The software in this package is designed to solve the problem

`max x'Qx`
`s.t. ||x||_0 <=k, ||x||_2 <=1`

using a branch-and-bound approach.  The code implements Optimal-SPCA in the paper "Certifiably Optimal Principal Component Analysis"  by Lauren Berk and Dimitris Bertsimas.


## Installation and set up

In order to run this software, you must install a recent version of Julia from http://julialang.org/downloads/.  The most recent version of Julia at the time this code was last tested was Julia 1.1.0.

Several packages must be installed in Julia before the code can be run.  These packages are Test, DataFrames, StatsBase, Printf, LinearAlgebra, JLD and Arpack.  They can be added by running:

`using Pkg`
`Pkg.add("Test")`
`Pkg.add("DataFrames")`
`Pkg.add("StatsBase")`
`Pkg.add("Printf")`
`Pkg.add("LinearAlgebra")`
`Pkg.add("JLD")`
`Pkg.add("Arpack")`

At this point, the files "test.jl", test1.jl" and test2.jl" should run successfully.  To run the script, navigate to the Algorithm directory and run:

`include("test.jl")`

or similar.

 The "test.jl" script will run Optimal-SPCA on the Pitprops dataset, and then generate an additional random problem and run the algorithm on that problem.  It will then identify the first few sparse principal components using Optimal-SPCA sequentially and reporting the cumulative variance explained. 
 
 The "test1.jl" script will reproduce the results generated for various datasets in the paper, and the "test2.jl" script will produce the optimal sparse principal component for different values of k in the pitprops dataset.

## Use of the branchAndBound() function

The key method in these packages is branchAndBound().  It takes two required  arguments: `prob`, and `k`.  The variable `prob` is a custom type that holds the original data as well as the covariance matrix associated with the problem.  (If data is not available, the cholesky factorization of the covariance matrix will suffice.)  The data is presented in an m x n array, with m  data points in n dimensions.  The corresponding covariance matrix is n x n.  The parameter `k` is a positive integer less than n.

By default, branchAndBound() solves the problem and returns the objective function value, solution vector, and  a few performance metrics, including time elapsed and the number of nodes explored. There are many optional parameters, some of which are discussed in detail in our paper. Other parameters have to do with technical aspects of the algorithm, like convergence criteria and resizing arrays.  These are commented on in  detail in the branchAndBound.jl file where the function is defined.


## Thank you

Thank you for your interest in Optimal-SPCA. Please let us know if you encounter any  issues using this code, or have comments or questions.  Feel free to email us anytime.

Lauren Berk
lberk@mit.edu

Dimitris Bertsimas
dbertsim@mit.edu
