# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 23:23:40 2021

@author: Daniel
"""

import numpy as np

def SparseReg(Theta, char_sizes=None, valid_single=None, opts=None):
# compute sparse regression on Theta * Xi = 0
# Theta: matrix of integrated terms
# char_sizes: vector of characteristic term sizes (per column)
# valid_single: vector of 1s/0s (valid single term model/not)
# opts: dictionary of options

    opt_defaults = {'threshold': 'threshold', 'brute_force': True, 'delta': 1e-15, 'epsilon': 1e-2, 'gamma': 3, 'verbose': False}

    # read options
    if opts is None:
        opts = dict() # to simplify conditional logic
    else:
        for opt in opt_defaults.keys():
            opt_value = opts[opt] if opts.has_key(opt) else opt_defaults[opt]
            exec(f'{opt}={opt_value}')
            
    h, w = Theta.shape
    
    if char_sizes is not None:
        for term in range(w):
            Theta[:, term] = Theta[:, term] / char_sizes[term] # renormalize by characteristic size
    if valid_single is None:
        valid_single = np.ones(shape=(w, 1))

    U, Sigma, V = np.linalg.svd(Theta, full_matrices=True)
    V = V.transpose() # since numpy SVD returns the transpose
    Xi = V[:, -1]
    if verbose:
        Sigmas = Sigma[Sigma[:]>0]
        print(V, np.log(Sigmas/np.min(Sigmas)))
    lambd = np.norm(Theta*Xi)
    if verbose:
        print(lambd)
    # find best one-term model as well
    for term in range(w):
        nrm[term] = np.norm(Theta[:, term]) / valid_single[term]
        lambda1, ind_single = min(nrm)
        if verbose:
            print(nrm[term])

    smallinds = np.zeros(shape=(w, 1))
    margins = np.zeros(shape=(w, 1)) # increases in residual per time step
    lambdas = np.zeros(shape=(w, 1))
    lambdas[0] = lambd
    if threshold != "multiplicative":
        Xis = np.zeros(shape=(w, 1)) # record coefficients
    for i in range(w):
        if threshold != "multiplicative":
            Xis[i] = Xi
        if brute_force:
           # product of the coefficient and characteristic size of library function
           res_inc = np.ones(shape=(w,1))*np.inf
        for p_ind in range(w):
            if brute_force:
                if smallinds[p_ind]==0:
                    # Try dropping each term
                    smallinds_copy = np.copy(smallinds)
                    smallinds_copy[p_ind] = 1
                    Xi_copy = Xi
                    Xi_copy[p_ind] = 0
                    _, _ , V = svd(Theta[:, smallinds_copy==0])
                    V = V.transpose()
                    Xi_copy[smallinds_copy==0] = V[:, -1]
                    res_inc[p_ind] = norm(Theta*Xi_copy)/lambd
            else:
                col = Theta[:, p_ind]
                # project out other columns
                for q_ind in range(w):
                    if (p_ind != q_ind) and smallinds[q_ind]==0:
                        other_col = Theta[:, q_ind]
                        col = col - np.dot(col, other_col)/np.norm(other_col)*other_col
                product[p_ind] = np.norm(Xi[p_ind]*col/np.norm(Theta))
                # product[p_ind] = np.norm(Xi[p_ind]*col)
        if brute_force:
            [Y, I] = min(res_inc)
            margins[i] = Y
            if verbose:
                print(res_inc)
                if threshold != "multiplicative":
                    print(i, lambd) # for pareto plot
            if (Y<=gamma) or (threshold != "multiplicative"):
                smallinds[I] = 1
                Xi[I] = 0
                [_, _, V] = svd(Theta[:, smallinds==0])
                Xi[smallinds==0] = V[:,end]
                lambd = norm(Theta*Xi)
                lambdas[i+1] = lambd
            if sum(smallinds==0)==1:
                    break
            else:
                print(Y, I)
                break
        else:
            product[smallinds==1]=np.inf
            [Y, I] = min(product)
            smallinds[I] = 1
            if sum(smallinds==0)==0:
                break
            if verbose:
                print(product.transpose())
            Xi_old = Xi
            Xi[smallinds==1] = 0 # set negligible terms to 0
            _ , _, V = svd(Theta[:,smallinds==0])
            V = V.transpose()
            Xi[smallinds==0] = V[:, -1]
            lambda_old = lambd
            lambd = norm(Theta*Xi)
            lambdas[i+1] = lambd
            margin = lambd/lambda_old
            if verbose:
                print(lambd, margin) 
            margins[i] = margin
            if (margin > gamma) and (lambd>delta) and (threshold=="multiplicative"):
                print(I)
                Xi = Xi_old
                print(Xi)
                break
            Xis[i+1] = Xi
            if threshold=="pareto":
                [Y_mar, I_mar] = max(margins)
                if n_terms>1:
                    I_mar = length(margins)-n_terms+1
                if verbose:
                    print(margins)
                    print(Y_mar, I_mar)
                stopping_point = I_mar-1;
                Xi = Xis[I_mar]; #stopping_point
                lambd = norm(Theta*Xi);
            elif threshold=="error":
                I_sm = np.argmax(lambdas>epsilon*lambda1)-1
                if isempty(I_sm):
                    I_sm = 0
                end
                Xi = Xis[I_sm] #stopping_point
                lambd = norm(Theta*Xi)

    # now compare single term and sparsified model
    if verbose:
        print(lambd, lambda1)
        print(lambdas)
    best_term = ind_single;
    if char_sizes is not None:
        Xi = Xi / char_sizes # renormalize by char. size
    return Xi, lambd, best_term, lambda1