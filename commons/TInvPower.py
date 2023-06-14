import numpy as np

def TInvPower(A, k, x0=None, mu0=None, tol=1e-12, max_iter=50, verbose=False):
    w = A.shape[0]
    if x0 is None:
        x = smallest_sv(A)
    else:
        x = x0
    if mu0 is None:
        mu = x.T @ A @ x
    else:
        mu = mu0
    if verbose:
        print("non-sparse mu: ", mu)
        print("non-sparse x: ", x)
    update_size = np.inf 
    it = 0
    while update_size>tol and it<max_iter:
        x /= np.linalg.norm(x)
        try:
            y = np.linalg.solve(A - mu * np.eye(w), x) # Rayleigh iteration
            # y = np.linalg.solve(A, x) # pure inverse iteration
        except:
            print("Exiting early due to singular matrix")
            return x, mu
        inds = np.argpartition(np.abs(y), -k)[-k:] # indices of largest absolute entries
        #y = keep_inds(y, inds)
        y = solve(A, inds) # solve subsystem with inds exactly
        y /= np.linalg.norm(y)
        #mu = y.T @ A @ y # comment out to fix mu
        update_size = min(np.linalg.norm(y - x), np.linalg.norm(y + x))/np.linalg.norm(y)
        if verbose:
            print("x:", x, "y:", y, "mu:", mu, "update_size:", update_size)
        x = y
        it += 1
    return x, mu, it

def keep_inds(vector, inds): # set all but inds of vector to 0
    temp = vector*0
    temp[inds] = vector[inds]
    return temp

def smallest_sv(A):
    U, Sigma, V = np.linalg.svd(A, full_matrices=True)
    V = V.transpose()  # since numpy SVD returns the transpose
    return V[:, -1] # smallest singular vector

def solve(A, inds):
    w = A.shape[1]
    x = np.zeros(shape=(w,))
    x[np.ix_(inds)] = smallest_sv(A[np.ix_(inds, inds)]) # work on submatrix with inds
    return x