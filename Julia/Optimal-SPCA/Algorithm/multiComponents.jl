using LinearAlgebra


function projectMatrix(sig, xVal)
	I = 1.0*Matrix(LinearAlgebra.I,size(sig,1),size(sig,1))
	xx = (I - xVal*xVal')
	A3 = xx*sig*xx
    return A3

end


function multiOptimalSPCA(prob, k, ncomp)
	myprob = problem(copy(prob.data),copy(prob.Sigma))
	st = time()
	all_x = zeros(ncomp, size(myprob.Sigma,1))

	for rnd = 1:ncomp
		~, xVal, ~, ~, ~, ~, ~ = branchAndBound(myprob, k, outputFlag=0, timeCap = 60) ;
		all_x[rnd,:] = xVal

		myprob.Sigma = projectMatrix(myprob.Sigma, xVal)
	end

	return all_x, time()-st
end


function adjVarExplained(Sigma, xVal)

	depth = size(xVal,1)
	varexplained = zeros(depth)
	sig = copy(Sigma)

	for component = 1:depth
		mypc = xVal[component,:]
		mypc = mypc/norm(mypc)
		varexplained[component] = mypc'*sig*mypc
		sig = projectMatrix(sig, mypc)
	end

	return varexplained
end
