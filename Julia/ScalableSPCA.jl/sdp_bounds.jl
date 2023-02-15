function getSDPUpperBound_gd(Sigma::Array{Float64, 2}, k::Int64, useSOCS::Bool=false, usePSDs::Bool=true)
# Note: greedy in terms of the rounding mechanism.

    n = size(Sigma, 1)
    max = maximum(Sigma)
    permissible_max = 1e12
    if max>permissible_max # rescale so Mosek doesn't complain
        scale = max/permissible_max
        Sigma = Sigma./scale
    else
        scale = 1
    end

    sdp_gd = Model(Mosek.Optimizer)
    if usePSDs
        @variable(sdp_gd, X[1:n, 1:n], PSD)
    else
        @variable(sdp_gd, X[1:n, 1:n], Symmetric)
        @constraint(sdp_gd, diagnonneg[i=1:n], X[i,i]>=0.0)
        @constraint(sdp_gd, twobytwominor[i=1:n, j=1:n], [X[i,i]+X[j,j]; X[i,i]-X[j,j]; 2.0*X[i,j]] in SecondOrderCone())
    end
    @variable(sdp_gd, z[1:n]>=0.0)
    @constraint(sdp_gd, z.<=1.0)

    @constraint(sdp_gd, sum(diag(X))==1.0)
    @constraint(sdp_gd, sum(z)<=k)
    @constraint(sdp_gd, diagX[i=1:n], X[i,i]<=z[i])
    @constraint(sdp_gd, offDiagX[i=1:n, j=1:n], (i!=j)*(X[i,j]-0.5*z[i])<=0.0)
    @constraint(sdp_gd, offDiagX2[i=1:n, j=1:n], (i!=j)*(-X[i,j]-0.5*z[i])<=0.0)
    if useSOCS
        @variable(sdp_gd, U[1:n, 1:n])
        @constraint(sdp_gd, sum(U)<=k)
        @constraint(sdp_gd, X.-U.<=0.0)
        @constraint(sdp_gd, X.+U.>=0.0)
        @constraint(sdp_gd, perspectiveRelaxation[i=1:n], [z[i]+X[i,i]; 2*X[:,i];z[i]-X[i,i]] in SecondOrderCone())
    end
    @objective(sdp_gd, Max, LinearAlgebra.dot(Sigma, X))

    @suppress optimize!(sdp_gd)

    #Get greedy solution and evaluate objective
    indices_gd=sortperm(value.(z), rev=true)[1:k]
    Sigma_reduced=Sigma[indices_gd, indices_gd]
    #if size(Sigma_reduced, 1)>3
    #    λs,xs=eigs(Sigma_reduced, nev=1)
    #    λ=real(λs[1])
    #    ind=1
    #else # eigs function complains if the matrix is too small
    λs,xs=eigen(Sigma_reduced)
    λ, ind = findmax(real.(λs)) # note: lambdas are negative!
    #end

    debug = false
    if debug
        println(Sigma)
        println(indices_gd)
        println(Sigma_reduced)
        println(λs,xs)
        println(λ, real.(xs[:, ind]))
        println(scale)
    end

    sol=zeros(n)
    sol[indices_gd]=real.(xs[:, ind]) # fill in the proper entries with eigenvector of reduced matrix
    return (JuMP.objective_value(sdp_gd)*scale), λ*scale, sol #added x return argument

end


function getSDPUpperBound_d2008_gd(Sigma::Array{Float64, 2}, k::Int64, useSOCS::Bool=false)
# Note: greedy in terms of the rounding mechanism.

    A=cholesky(Sigma) #Need to use A.U
    n = size(Sigma, 1)

    sdp_gd = Model(SCS.Optimizer)
    if n<=13
        sdp_gd=Model(Mosek.Optimizer)
    end


    @variable(sdp_gd, X[1:n, 1:n], PSD)
    @variable(sdp_gd, P[1:n, 1:n, 1:n]) #let third dimension index each matrix
    @constraint(sdp_gd, imposePSDConstraint[i=1:n], P[:,:,i] in PSDCone())
    @constraint(sdp_gd, imposePSDConstraint2[i=1:n], X.-P[:,:,i] in PSDCone())

    @variable(sdp_gd, z[1:n]>=0.0)
    @constraint(sdp_gd, z.<=1.0)

    @constraint(sdp_gd, sum(diag(X))==1.0)
    @constraint(sdp_gd, imposebinaries[i=1:n], sum(P[j,j,i] for j=1:n)==z[i])
    if useSOCS
        @variable(sdp_gd, U[1:n, 1:n])
        @constraint(sdp_gd, sum(U)<=k)
        @constraint(sdp_gd, X.-U.<=0.0)
        @constraint(sdp_gd, X.+U.>=0.0)
        @constraint(sdp_gd, perspectiveRelaxation[i=1:n], [z[i]+X[i,i]; 2*X[:,i];z[i]-X[i,i]] in SecondOrderCone())
    end

    @objective(sdp_gd, Max, sum(LinearAlgebra.dot(A.U[i,:]*A.U[i,:]', P[:,:,i]) for i=1:n))
    #@show "About to optimize!"
    @suppress optimize!(sdp_gd)
    #@show getobjectivevalue(sdp_gd)

    #Get greedy solution and evaluate objective
    indices_gd=sortperm(value.(z), rev=true)[1:k]
    Sigma_reduced=Sigma[indices_gd, indices_gd]
    λs,xs=eigs(Sigma_reduced, nev=1)
    λ=λs[1]

    return (JuMP.objective_value(sdp_gd)), λ
end

function getSDPUpperBound_gd_highdim(Sigma::Array{Float64, 2}, k::Int64, useSOCS::Bool=false, useCuts=false)
# Note: greedy in terms of the rounding mechanism.

    n = size(Sigma, 1)
    max = maximum(Sigma)
    permissible_max = 1e12
    if max>permissible_max # rescale so Mosek doesn't complain
        scale = max/permissible_max
        Sigma = Sigma./scale
    else
        scale = 1
    end

    sdp_gd = Model(Mosek.Optimizer)

    @variable(sdp_gd, X[1:n, 1:n], Symmetric)
    @variable(sdp_gd, z[1:n]>=0.0)
    @constraint(sdp_gd, z.<=1.0)

    @constraint(sdp_gd, sum(diag(X))==1.0)
    @constraint(sdp_gd, sum(z)<=k)
    @constraint(sdp_gd, diagX[i=1:n], X[i,i]<=z[i])
    @constraint(sdp_gd, offDiagX[i=1:n, j=1:n], (i!=j)*(X[i,j]-0.5*z[i])<=0.0)
    @constraint(sdp_gd, offDiagX2[i=1:n, j=1:n], (i!=j)*(-X[i,j]-0.5*z[i])<=0.0)
    if useSOCS
        @variable(sdp_gd, U[1:n, 1:n])
        @constraint(sdp_gd, sum(U)<=k)
        @constraint(sdp_gd, X.-U.<=0.0)
        @constraint(sdp_gd, X.+U.>=0.0)
        @constraint(sdp_gd, perspectiveRelaxation[i=1:n], [z[i]+X[i,i]; 2*X[:,i];z[i]-X[i,i]] in SecondOrderCone())
    end


    @objective(sdp_gd, Max, LinearAlgebra.dot(Sigma, X))

    t_start=time()
    @suppress optimize!(sdp_gd)
    #getobjectivevalue(sdp_gd)

    if useCuts
     j=0
     maxCuts=20
         while j<maxCuts
             λ, ϕ=eigs(value.(X),  nev=1, which=:SR)
                if real(λ[1])<=-1e-4
                     cutV=real(ϕ[:, 1])
                     @constraint(sdp_gd, LinearAlgebra.dot(cutV*cutV', X)>=0.0)
                else
                    break
                end
             @suppress optimize!(sdp_gd)
             j+=1
        end
    end

    #Get greedy solution and evaluate objective
    indices_gd=sortperm(value.(z), rev=true)[1:k]
    Sigma_reduced=Sigma[indices_gd, indices_gd]
    #λs,xs=eigs(Sigma_reduced, nev=1)
    #λ=λs[1]

    λs,xs=eigen(Sigma_reduced)
    λ, ind = findmax(real.(λs)) # note: lambdas are negative!
    #end

    sol=zeros(n)
    sol[indices_gd]=real.(xs[:, ind]) # fill in the proper entries with eigenvector of reduced matrix
    return (JuMP.objective_value(sdp_gd)*scale), λ*scale, sol #added x return argument

    #return (getobjectivevalue(sdp_gd)), λ
end
