import numpy as np

def sparse_reg(Theta, opts=None, threshold='pareto', brute_force=True, delta=1e-10, epsilon=1e-2, gamma=2,
               verbose=False, n_terms=-1, char_sizes=None, row_norms=None, valid_single=None, avoid=[], subinds=None, anchor_col=0):
# compute sparse regression on Theta * Xi = 0
# Theta: matrix of integrated terms
# char_sizes: vector of characteristic term sizes (per column)
# row_norms: desired norm of each row
# valid_single: vector of 1s/0s (valid single term model/not)
# opts: dictionary of options
# avoid: coefficient vectors to be orthogonal to
# and a lot more not described above
    
    Theta = np.copy(Theta) # avoid bugs where array is modified in place
    if row_norms is not None:
        for row in range(len(row_norms)):
            #rownm = np.linalg.norm(Theta[row, :])
            #if rownm != 0:
            #    Theta[row, :] *= (row_norms[row]/rownm)
            Theta[row, :] *= row_norms[row]
    if char_sizes is not None: # do this here: char_sizes are indexed by full column set
        char_sizes = np.array(char_sizes)
        char_sizes /= np.max(char_sizes)
        for term in range(len(char_sizes)):
            Theta[:, term] = Theta[:, term] / char_sizes[term] # renormalize by characteristic size
    if anchor_col is None: # do this exactly here: when we divide by Thetanm later, we work with the normalized columns
        Thetanm = np.linalg.norm(Theta)
    else: # do this here: anchor_col is also indexed by full columns set
        Thetanm = np.linalg.norm(Theta[:, anchor_col])
    
    if subinds is not None:
        if subinds == []: # no inds allowed at all
            return None, np.inf, None, np.inf
        Theta = Theta[:, subinds]
        if char_sizes is not None:
            char_sizes = np.array(char_sizes)
            char_sizes = char_sizes[subinds]
        if valid_single is not None:
            valid_single = np.array(valid_single)
            valid_single = valid_single[subinds]
    M = 100*Theta.shape[0]
    for Xi in avoid:
        Theta = np.vstack([Theta, M*np.transpose(Xi)]) # acts as a constraint - weights should be orthogonal to Xi
    
    h, w = Theta.shape
    if anchor_col is None:
        Thetanm /= np.sqrt(w) # scale norm of Theta by square root of # columns to fix scaling of Theta@Xi vs Thetanm
    beta = w/h # aspect ratio
    
    if valid_single is None:
        valid_single = np.ones(shape=(w, 1))

    U, Sigma, V = np.linalg.svd(Theta, full_matrices=True)
    V = V.transpose() # since numpy SVD returns the transpose
    Xi = V[:, -1]
    if verbose:
        #print("Sigma:", Sigma)
        #Sigmas = Sigma[Sigma[:]>0]
        Sigma_shrink = [opt_shrinker(s, beta) for s in Sigma]
        #print("Sigma_shrink:", Sigma_shrink)
        print("V:", V)
        print("scores:", np.log(Sigma)) # np.log(Sigmas)/np.min(Sigmas)
    lambd = np.linalg.norm(Theta@Xi)/Thetanm
    if verbose:
        print('lambda:', lambd)
    # find best one-term model as well
    nrm = np.zeros(w)
    for term in range(w):
        nrm[term] = np.linalg.norm(Theta[:, term]) / (Thetanm*valid_single[term])
        lambda1, best_term = min(nrm), np.argmin(nrm)
        if verbose:
            print(f'nrm[{term}]:', nrm[term])
    if w==1: # no regression to run
        return None, np.inf, best_term, lambda1
    
    smallinds = np.zeros(w)
    margins = np.zeros(w) # increases in residual per time step
    lambdas = np.zeros(w)
    lambdas[0] = lambd
    Xis = np.zeros(shape=(w, w)) # record coefficients
    flag = False
    for i in range(w-1):
        Xis[i] = Xi
        if brute_force:
           # product of the coefficient and characteristic size of library function
           res_inc = np.ones(shape=(w,1))*np.inf
        product = np.zeros(shape=(w,1))
        for p_ind in range(w):
            if brute_force:
                if smallinds[p_ind]==0:
                    # Try dropping each term
                    smallinds_copy = np.copy(smallinds)
                    smallinds_copy[p_ind] = 1
                    Xi_copy = np.copy(Xi)
                    Xi_copy[p_ind] = 0
                    _, _, V = np.linalg.svd(Theta[:, smallinds_copy==0], full_matrices=True)
                    V = V.transpose()
                    Xi_copy[smallinds_copy==0] = V[:, -1]
                    res_inc[p_ind] = np.linalg.norm(Theta@Xi_copy)/Thetanm/lambd
            else:
                col = Theta[:, p_ind]
                # project out other columns
                for q_ind in range(w):
                    if (p_ind != q_ind) and smallinds[q_ind]==0:
                        other_col = Theta[:, q_ind]
                        col = col - np.dot(col, other_col)/np.linalg.norm(other_col)**2*other_col
                #product[p_ind] = np.linalg.norm(Xi[p_ind]*col)/np.linalg.norm(Theta)
                product[p_ind] = np.linalg.norm(Xi[p_ind]*col)
        if brute_force:
            Y, I = min(res_inc), np.argmin(res_inc)
            if verbose:
                print("Y:", Y, "I:", I)
            margins[i] = Y
            if verbose:
                #print('res_inc:', res_inc)
                if threshold != "multiplicative":
                    print("i", i, "lambda", lambd) # for pareto plot
            if (Y<=gamma) or (threshold != "threshold") or (lambd<=delta):
                smallinds[I] = 1
                Xi[I] = 0
                _, _, V = np.linalg.svd(Theta[:, smallinds==0], full_matrices=True)
                V = V.transpose()
                Xi[smallinds==0] = V[:, -1]
                lambd = np.linalg.norm(Theta@Xi)/Thetanm
                lambdas[i+1] = lambd
                if sum(smallinds==0)==1:
                        margins[-1] = np.inf
                        break
            else:
                if verbose:
                    print("Y:", Y, "I:", I)
                break
        else:
            product[smallinds==1]=np.inf
            Y, I = min(product), np.argmin(product)
            if verbose:
                print("Y:", Y, "I:", I)
            smallinds[I] = 1
            if sum(smallinds==0)==1:
                break
            #if verbose:
                #print("prod:", product.transpose())
            Xi_old = Xi
            Xi[smallinds==1] = 0 # set negligible terms to 0
            _ , _, V = np.linalg.svd(Theta[:,smallinds==0], full_matrices=True)
            V = V.transpose()
            Xi[smallinds==0] = V[:, -1]
            lambda_old = lambd
            lambd = np.linalg.norm(Theta@Xi)/Thetanm
            lambdas[i+1] = lambd
            margin = lambd/lambda_old
            if verbose:
                print("lambda:", lambd, " margin:", margin) 
            margins[i] = margin
            if (margin > gamma) and (lambd>delta) and (threshold=="threshold"):
                print("I:", I)
                Xi = Xi_old
                print("Xi:", Xi)
                break
    Xis[i+1] = Xi
    if threshold=="pareto":
        Y_mar, I_mar = max(margins), np.argmax(margins)
        if n_terms>1:
            I_mar = sum(margins>0)-n_terms
        if verbose:
            print("margins:", margins)
            print("Y_mar:", Y_mar, "I_mar:", I_mar)
        I_mar = max(np.argmax(lambdas>delta)-1, I_mar)
        stopping_point = I_mar-1
        Xi = Xis[I_mar] #stopping_point
        lambd = np.linalg.norm(Theta@Xi)/Thetanm
    elif threshold=="multiplicative":
        I_sm = np.argmax((lambdas>epsilon*lambda1) & (lambdas>delta))-1
        Xi = Xis[I_sm] #stopping_point
        lambd = np.linalg.norm(Theta@Xi)/Thetanm
    else:
        if n_terms>1: ### Don't think this line does anything functionally but I also don't really use this
            I_mar = sum(margins>0)-n_terms
        lambdas[0] = lambdas[1] ### DUCT TAPE since I don't know what's going on (basically first lambda is big)
        I_mar = max(np.argmax(lambdas>delta)-1, np.argmax(margins>gamma))
        if verbose:
            print(lambdas>delta)
            print(margins==0)
            print("I_mar:", I_mar)
        Xi = Xis[I_mar] #stopping_point
        lambd = np.linalg.norm(Theta@Xi)/Thetanm

    if verbose:
        print("Xis:", Xis)
    # now compare single term and sparsified model
    if verbose:
        print("lambda:", lambd, "lambda1:", lambda1)
        print("lambdas:", lambdas)
    #print("Xi1:", Xi)
    if char_sizes is not None:
        Xi = Xi / char_sizes # renormalize by char. size
        #print("Xi2:", Xi)
        #nm = np.linalg.norm(Xi)
        ## divide errors by the norm to make errors consistent
        #lambd /= nm
        #lambda1 /= nm
    #divide errors by square root of number of rows to make errors consistent
    #lambd /= h**0.5
    #lambda1 /= h**0.5
    if -min(Xi)>max(Xi): # ensure vectors are "positive"
        Xi = -Xi
    Xi = Xi/max(Xi) # make largest coeff 1
    # make residuals relative to original norm(Theta)*norm(Xi)
    nm = np.linalg.norm(Xi)
    #lambd /= (nm*Thetanm)
    #lambd /= Thetanm
    #lambda1 /= Thetanm
    
    return Xi, lambd, best_term, lambda1

def opt_shrinker(y, beta):
    if y <= 1+np.sqrt(beta):
        return 0
    else:
        return np.sqrt((y*y-beta-1)**2-4*beta)/y
    
def regress(Theta, col_numbers): # regression on a fixed set of terms
    h, w = Theta.shape
    #Thetanm = np.linalg.norm(Theta)
    Thetanm = np.linalg.norm(Theta[:, 0])
    # fix scaling w/ respect to number of columns
    Thetanm /= np.sqrt(w)
    smallinds = np.ones(shape=(w,))
    Xi = np.zeros(shape=(w,))
    smallinds[np.array(col_numbers)] = 0
    _, _, V = np.linalg.svd(Theta[:, smallinds==0], full_matrices=True)
    V = V.transpose()
    Xi[smallinds==0] = V[:, -1]
    lambd = np.linalg.norm(Theta@Xi)
    if -min(Xi)>max(Xi): # ensure vectors are "positive"
        Xi = -Xi
    Xi = Xi/max(Xi) # make largest coeff 1
    # make residuals relative to original norm(Theta)*norm(Xi)
    nm = np.linalg.norm(Xi)
    #lambd /= (nm*Thetanm)
    lambd /= Thetanm
    return Xi, lambd