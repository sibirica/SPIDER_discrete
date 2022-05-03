import numpy as np
from scipy.ndimage import gaussian_filter

# probably won't be used
def coarse_grain(binned, sigma, truncate=8, wrap=True):
    sigma_vector = np.zeros(shape=(len(binned.shape),1))
    sigma_vector[0:2] = sigma
    if wrap:
        smoothed = gaussian_filter(binned, sigma_vector, mode='wrap', truncate=truncate)
    else:
        smoothed = gaussian_filter(binned, sigma_vector, truncate=truncate)
    return smoothed

# consider using square-cubed kernel which has 3 derivatives vanishing on edges (see Wikipedia KDE)
def gauss1d(x0, sigma, truncate=8, xmin=-np.inf, xmax=np.inf, wrap=True):
    # truncate = 6 or 8 should be fine
    offset = int(x0)
    s22 = 2*sigma*sigma
    halfwidth = int(np.ceil(truncate*sigma))
    if not wrap:
        mn = np.max([offset-halfwidth, xmin])
        mx = np.min([offset+halfwidth, xmax])+1e-6 # so that range includes mx
    else:
        mn = offset-halfwidth 
        mx = offset+halfwidth+1e-6
    x = np.arange(mn, mx)
    gx = np.exp(-(x-x0)*(x-x0)/(s22))
    #gx /= ((2*np.pi)**0.5*sigma) # (normalize approximately)
    gx /= sum(gx) 
    return gx, mn, mx