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
# n_terms: if not -1, select from models with this number of terms and below
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
        if not subinds:  # no inds allowed at all
            return None, np.inf, None, np.inf
        theta = theta[:, subinds]
        if char_sizes is not None:
            char_sizes = np.array(char_sizes)
            char_sizes = char_sizes[subinds]
        if valid_single is not None:
            valid_single = np.array(valid_single)
            valid_single = valid_single[subinds]        
            
    # functionality probably outdated with more advanced model selection code
    M = 100*Theta.shape[0]
    for Xi in avoid:
        Theta = np.vstack([Theta, M*np.transpose(Xi)]) # acts as a constraint - weights should be orthogonal to Xi
    
    h, w = Theta.shape
    if anchor_col is None:
        Thetanm /= np.sqrt(w) # scale norm of Theta by square root of # columns to fix scaling of Theta@Xi vs Thetanm
    #beta = w/h # aspect ratio
    
    if valid_single is None:
        valid_single = np.ones(shape=(w, 1))
            
    U, Sigma, V = np.linalg.svd(Theta, full_matrices=True)
    V = V.transpose() # since numpy SVD returns the transpose
    Xi = V[:, -1]
    if verbose:
        pass
        # print("sigma:", sigma)
        # Sigmas = sigma[sigma[:]>0]
        # Sigma_shrink = [opt_shrinker(s, beta) for s in sigma]
        # print("Sigma_shrink:", Sigma_shrink)
        # print("v:", v)
        # print("scores:", np.log(sigma)) # np.log(Sigmas)/np.min(Sigmas)
    lambd = np.linalg.norm(theta @ xi) / thetanm
    if verbose:
        print('lambda:', lambd)
    # find best one-term model
    nrm = np.zeros(w)
    for term in range(w):
        nrm[term] = np.linalg.norm(theta[:, term]) / (thetanm * valid_single[term])
        lambda1, best_term = min(nrm), np.argmin(nrm)
        if verbose:
            pass
            # print(f'nrm[{term}]:', nrm[term])
    if w == 1:  # no regression to run
        # noinspection PyUnboundLocalVariable
        return None, np.inf, best_term, lambda1

    smallinds = np.zeros(w)
    margins = np.zeros(w)  # increases in residual per time step
    lambdas = np.zeros(w)
    lambdas[0] = lambd
    xis = np.zeros(shape=(w, w))  # record coefficients

    for i in range(w - 1):
        xis[i] = xi
        if brute_force:
            # product of the coefficient and characteristic size of library function
            res_inc = np.ones(shape=(w, 1)) * np.inf
        product = np.zeros(shape=(w, 1))
        for p_ind in range(w):
            if brute_force:
                if smallinds[p_ind] == 0:
                    # Try dropping each term
                    smallinds_copy = np.copy(smallinds)
                    smallinds_copy[p_ind] = 1
                    xi_copy = np.copy(xi)
                    xi_copy[p_ind] = 0
                    _, _, v = np.linalg.svd(theta[:, smallinds_copy == 0], full_matrices=True)
                    v = v.transpose()
                    xi_copy[smallinds_copy == 0] = v[:, -1]
                    # noinspection PyUnboundLocalVariable
                    res_inc[p_ind] = np.linalg.norm(theta @ xi_copy) / thetanm / lambd
            else:
                col = theta[:, p_ind]
                # project out other columns
                for q_ind in range(w):
                    if (p_ind != q_ind) and smallinds[q_ind] == 0:
                        other_col = theta[:, q_ind]
                        col = col - np.dot(col, other_col) / np.linalg.norm(other_col) ** 2 * other_col
                # product[p_ind] = np.linalg.norm(xi[p_ind]*col)/np.linalg.norm(Theta)
                product[p_ind] = np.linalg.norm(xi[p_ind] * col)
        if brute_force:
            y, ii = min(res_inc), np.argmin(res_inc)
        else:
            product[smallinds == 1] = np.inf
            y, ii = min(product), np.argmin(product)
        if verbose:
            print("y:", y, "ii:", ii)
        smallinds[ii] = 1
        xi[ii] = 0
        _, _, v = np.linalg.svd(theta[:, smallinds == 0], full_matrices=True)
        v = v.transpose()
        xi[smallinds == 0] = v[:, -1]
        lambda_old = lambd
        lambd = np.linalg.norm(theta @ xi) / thetanm
        lambdas[i + 1] = lambd
        if brute_force:
            margins[i] = y
        else:
            margin = lambd / lambda_old
            margins[i] = margin
        if sum(smallinds == 0) == 1:
            margins[-1] = np.inf
            break
    # fill in last xi
    xis[w] = xi
    # decision rules
    if threshold == "pareto":
        y_mar, i_mar = max(margins), np.argmax(margins)
        if verbose:
            print('y_mar:', y_mar)
        i_mar = max(np.argmax(lambdas > delta) - 1, i_mar)
        if verbose:
            print("margins:", margins)
    elif threshold == "multiplicative":
        # noinspection PyUnboundLocalVariable
        i_mar = np.argmax((lambdas > epsilon * lambda1) & (lambdas > delta)) - 1
    else:
        # lambdas[0] = lambdas[1] ### DUCT TAPE - maybe numerical error in svd?
        i_mar = max(np.argmax(lambdas > delta) - 1, np.argmax(margins > gamma))
    if verbose:
        print("i_mar:", i_mar)
    if n_terms > 0:
        ind_n_terms = w - n_terms
        # if more than n_terms, advance to next index
        i_mar = max(i_mar, ind_n_terms)
    xi = xis[i_mar]
    lambd = np.linalg.norm(theta @ xi) / thetanm

    if verbose:
        print("xis:", xis)
        print("lambda:", lambd, "lambda1:", lambda1)
        print("lambdas:", lambdas)
    if char_sizes is not None:
        xi = xi / char_sizes  # renormalize by char. size
    # divide errors by square root of number of rows to make errors consistent
    # lambd /= h**0.5
    # lambda1 /= h**0.5
    if -min(xi) > max(xi):  # ensure vectors are "positive"
        xi = -xi
    if max(xi) > 0:
        xi = xi / max(xi)  # make largest coeff 1

    # noinspection PyUnboundLocalVariable
    return xi, lambd, best_term, lambda1


# was never properly implemented and probably never will be
def opt_shrinker(y, beta):
    if y <= 1 + np.sqrt(beta):
        return 0
    else:
        return np.sqrt((y * y - beta - 1) ** 2 - 4 * beta) / y


def regress(Theta, col_numbers):  # regression on a fixed set of terms
    h, w = Theta.shape
    #Thetanm = np.linalg.norm(Theta)
    Thetanm = np.linalg.norm(Theta[:, 0])
    smallinds = np.ones(shape=(w,))
    xi = np.zeros(shape=(w,))
    smallinds[np.array(col_numbers)] = 0
    _, _, v = np.linalg.svd(Theta[:, smallinds == 0], full_matrices=True)
    v = v.transpose()
    xi[smallinds == 0] = v[:, -1]
    lambd = np.linalg.norm(Theta @ xi)
    if -min(xi) > max(xi):  # ensure vectors are "positive"
        xi = -xi
    xi = xi / max(xi)  # make largest coeff 1
    # make residuals relative to original norm(Theta)*norm(xi)
    nm = np.linalg.norm(xi)
    # lambd /= (nm*thetanm)
    lambd /= thetanm
    return xi, lambd
