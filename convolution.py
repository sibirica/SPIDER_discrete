import numpy as np
from scipy.ndimage import gaussian_filter

def coarse_grain(binned, sigma, truncate=12):
    sigma_vector = np.zeros(shape=(len(binned.shape),1))
    sigma_vector[0:2] = sigma
    smoothed = gaussian_filter(binned, sigma_vector, mode='wrap', truncate=truncate)
    return smoothed

def gauss1d(x0, sigma, truncate=12, xmin=0, xmax=0, wrap=True):
    # truncate = 6 or 8 should be fine but...
    offset = int(x0)
    halfwidth = int(np.ceil(truncate*sigma))
    if not wrap:
        mn = np.max([offset-halfwidth, xmin])
        mx = np.min([offset+halfwidth, xmax])
    else:
        mn = offset-halfwidth 
        mx = offset+halfwidth
    x = np.arange(mn, mx)
    gx = np.exp(-(x-x0)**2/(2*sigma**2))
    #gx /= ((2*np.pi)**0.5*sigma) # normalize
    gx /= sum(gx) 
    return gx, mn, mx