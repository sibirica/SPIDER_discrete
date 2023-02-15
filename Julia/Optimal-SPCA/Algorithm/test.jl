# Required packages
using DataFrames, StatsBase, Printf, LinearAlgebra

# Reading in source files
include("dat.jl");
include("branchAndBound.jl");
include("utilities.jl");
include("multiComponents.jl");




# Pitprops example
k=5; # desired sparsity
prob = pitprops; #problem selection
obj, xVal, timetoBound, timetoConverge, timeOut,
	explored, toPrint = branchAndBound(prob, k) ;

print(@sprintf("objective value: %.4e\n", obj));
print(@sprintf("time to best solution: %.4e\n", timetoBound));
print(@sprintf("time to converge: %.4e\n", timetoConverge));
print(@sprintf("nodes explored: %d\n", explored));
println("solution vector:");
print(xVal);
println("\n");

# A synthetic example
dataset = randn(20,100);
Sigma = dataset'*dataset/20;
syntheticprob = problem(dataset, Sigma);

k=5; # desired sparsity
prob = syntheticprob; #problem selection
obj, xVal, timetoBound, timetoConverge, timeOut,
	explored, toPrint = branchAndBound(prob, k, outputFlag=0) ;

print(@sprintf("objective value: %.4e\n", obj));
print(@sprintf("time to best solution: %.4e\n", timetoBound));
print(@sprintf("time to converge: %.4e\n", timetoConverge));
print(@sprintf("nodes explored: %d\n", explored));
println("solution vector:");
print(xVal);
println("\n");

# Multiple Components
k=5; # desired sparsity
prob = pitprops; #problem selection
ncomp = 4;
all_x, timeOut = multiOptimalSPCA(prob, k, ncomp);

total_variance = adjVarExplained(prob.Sigma, all_x)

println("adjusted variance explained:");
print(total_variance);
print(@sprintf("total time: %.4e\n", timeOut));
println("solution vectors:");
print(all_x);
