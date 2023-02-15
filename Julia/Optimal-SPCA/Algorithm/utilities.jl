using DataFrames, StatsBase

mutable struct problem
    data::Array{Float64}
    Sigma::Array{Float64}
end

# return obejctive and solution given a solution vector
function evaluate(solution, prob)
	return (transpose(solution)*prob.Sigma*solution)[1], solution
end

# return objective and solution given a support vector (of 0s and 1s)
function evalSupport(support, prob)
	xVal = LinearAlgebra.eig(Qhat(prob.Sigma,support))[2][:,size(prob.Sigma,1)]
	return evaluate(xVal, prob)
end

# return objective and solution given the indices of a support
function evalIndices(support, prob)
	y = round.(Int64,zeros(size(prob.Sigma,1)))
	y[support[support.!=0]] = 1
	xVal = LinearAlgebra.eig(Qhat(prob.Sigma,y))[2][:,size(prob.Sigma,1)]
	return evaluate(xVal, prob)
end

# routine for creating a new problem out of a large data set by selecting
# a subset of dimensions out of all the variables in the problem
function shrinkProblem(prob, dimensions)
	finished = false
	problem2 = problem(prob.data, prob.Sigma);
	while !finished
		problem2 = problem(prob.data, prob.Sigma);
		r = problem2.data
		if size(r)[2]>dimensions
			selection = sample(1:size(r)[2] ,dimensions, replace=false)
			r = r[:, selection]
		end
		r = r .- Transpose(mean.(eachcol(r)))
		problem2.data = r
		problem2.Sigma = r'*r/size(r)[1];
		# finished = isposdef(problem2.Sigma) # removing so we can go higher than 250
		finished = true
	end
	return problem2
end


# zeroes out rows and columns from mat that correspond to support = 0
function Qhat(mat, support)
	yDel = (support.==0)
	Q = copy(mat)
	Q[yDel , : ] = 0
	Q[ : , yDel] = 0
	return Q
end

#Faster computation of maximum eigenvalue and eigenvector
function myeigmax(SigOrA, thisStart, y, highDim, refineCap)
	yKeep = .!(y.==0)
	m=0
	if highDim
		thisA = copy(SigOrA[:, yKeep])
		m, n = size(thisA)
		Q = thisA*thisA'/(m-1)
		b = thisStart[yKeep]
		b = thisA*(b / LinearAlgebra.LinearAlgebra.norm(b))
		normb = 0
		newnorm = LinearAlgebra.norm(b)
	else
		Q = copy(SigOrA[yKeep,yKeep])
		beta0 = thisStart[yKeep]
		normb = LinearAlgebra.norm(beta0)
		b = Q*beta0
		newnorm = LinearAlgebra.norm(b)
	end

	cycles = 0
	while abs(normb - newnorm) > .0001*normb && cycles < 100
		normb = newnorm
		b = Q*(b / normb)
		newnorm = LinearAlgebra.norm(b)
		if refineCap>0
			if newnorm > refineCap
				return refineCap, copy(b/normb)
			end
		end
		cycles = cycles + 1
	end

	if highDim
		b = thisA'*b
		b = thisA'*(thisA*b/LinearAlgebra.norm(b))/(m-1)
		normb = LinearAlgebra.norm(b)
	end

	b = b/normb
	expandedb = zeros(length(y))
	expandedb[yKeep]=b

	return normb, expandedb
end

#Creates a sparse version of a vector by killing smallest components and scaling up
function Hk(origlist, sparsity, support)
	list = real(copy(origlist))
	ksparse = zeros(length(list))
	indicesToKeep = (support.==1)
	dummyvalue = minimum(list)-1
	list[(support.>-1)] .= dummyvalue

	newIndices = selectperm2(list, sparsity-sum(indicesToKeep))
	indicesToKeep[newIndices].=true

	ksparse[indicesToKeep]=origlist[indicesToKeep]

	ksparse = ksparse / LinearAlgebra.norm(ksparse[indicesToKeep])

	return ksparse
end

# selects *indices* of x that correspond to the k largest values of x
# for use in Hk utility
function selectperm2(x,k)
    if k > 1
        kk = 1:k
    else
        kk = 1
    end
    z = collect(1:length(x))
    return partialsort!(z,1:k,by = (i)->abs(x[i]), rev = true)
end

# selects out of x the k largest values
# for use in B+B algorithm
function selectsorted(x,k)
    if k > 1
        kk = 1:k
    else
        kk = 1
    end
    z = collect(1:length(x))
    return partialsort!(x,1:k,by = (i)->(i), rev = true)
end

# partitions a vector of length x into n parts
yourpart(x,n) = Any[Any[x[i:n:length(x)]] for i in 1:n]

# faster matrix vector multiplication when the vector is sparse
function sparseTimes(mat, vec)
    ind = abs.(vec).>.000001
    prod = mat[:,ind]*vec[ind]
    return prod
end

# Yuan and zhang algorithm with random restarts
function subset(prob, k; timeLimit = 7200, support = zeros(1), countdown = 100)
	n = size(prob.Sigma,1)
	lambdas, betas, = KrylovKit.eigsolve(prob.Sigma)
	betas = reduce(hcat, betas) # need matrix instead of vector of vectors
	#betas = reduce(vcat,transpose.(betas)) 
	#lambdas, betas, =Arpack.eigs(prob.Sigma, nev=1, which=:LR)
	beta0=betas[:,1]
	if length(support)==1
		support = zeros(n).-1
	end

	#Starts with local search starting at first eigenvector
	bestBeta = eigSubset(prob, support, k, beta0)
	bestObj, ~ = evaluate(bestBeta, prob)
	start = time()
	#Tries a thousand random other starting points
	margin = 1
	while countdown > 0 && time()-start < timeLimit
		beta = rand(size(prob.Sigma,1))
		beta = beta / LinearAlgebra.norm(beta0)
		beta = eigSubset(prob, support, k, beta)
		obj, ~ = evaluate(beta, prob)
		if obj > bestObj
			bestObj = obj
			bestBeta = copy(beta)
			countdown = 100
		end
		countdown = countdown - 1
	end

	return bestObj, bestBeta
end

#Applies Yuan and Zhang heuristic starting from first eigenvector
function eigSubset(prob, support, k, beta0)
	beta = Hk(beta0, k, support)
	for i=1:100
		beta = Hk(prob.Sigma*beta, k, support)
	end
	return beta
end

# Creates a new file at filename with specified header
function refreshFile(filename, header)
	outfile = open(filename, "w");
	close(outfile);
	outfile = open(filename, "a");
	write(outfile,header);
	close(outfile);
end
