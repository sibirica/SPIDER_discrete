import numpy as np
from numba import jit, float64, uint64, prange
from numba_kdtree import KDTree


@jit(
    signature_or_function="float64[:](float64[:, :], float64[:], float64[:, :], float64)",
    nopython=True,
    cache=True,
    fastmath=False,
    parallel=True,
    nogil=True)
def gaussian_coarse_grain2d(points: float64[:, :],
                            values: float64[:],
                            xi: float64[:, :],
                            sigma: float64) -> float64[:]:
    """
    This function implements a gaussian coarse graining algorithm. Heavily inspired by the scipy implementation at
    https://github.com/scipy/scipy/blob/main/scipy/stats/_stats.pyx
    :param points: the data points to estimate from in 2 dimensions. Shape (n, 2).
    :param values: the multivariate values associated with the data points. (n,)
    :param xi: the coordinates to evaluate the estimate at in 2 dimensions. Shape (m, 2).
    :param sigma: the gaussian kernel width (standard deviation). Float.
    :return: the coarse grained data at the coordinates xi. Shape (m,).
    """
    n: uint64 = points.shape[0]  # number of data points
    d: uint64 = 2  # dimension of the data points
    m: uint64 = xi.shape[0]  # number of evaluation points

    points_: float64[n, d] = points / sigma  # the scaled data points
    xi_: float64[m, d] = xi / sigma  # the scaled evaluation points

    estimate: float64[m] = np.zeros(m)  # the estimate at the evaluation points
    norm: float64 = 1 / (2 * np.pi) / (sigma * sigma)  # the normalization factor of the gaussian kernel

    for i in prange(n):
        local_estimate: float64[m] = np.zeros(m)  # intermediate results for each thread
        for j in range(m):
            local_estimate[j] += np.exp(-np.sum((points_[i, :] - xi_[j, :]) * (points_[i, :] - xi_[j, :])) / 2) \
                                 * values[i]

        # Reduction operation to combine the intermediate results
        for j in prange(m):
            estimate[j] += local_estimate[j]

    return estimate * norm


@jit(
    signature_or_function="float64[:](float64[:, :], float64[:], float64[:, :], float64)",
    nopython=True,
    cache=False,
    fastmath=False,
    parallel=True,
    debug=False,
    nogil=True,
    boundscheck=True
)
def kd_gaussian_coarse_grain2d(points: float64[:, :],
                               values: float64[:],
                               xi: float64[:, :],
                               sigma: float64) -> float64[:]:
    """
    This function implements a gaussian coarse graining algorithm. Uses a KDTree to only consider nearby points.
    Heavily inspired by the scipy implementation at https://github.com/scipy/scipy/blob/main/scipy/stats/_stats.pyx
    :param points: the data points to estimate from in 2 dimensions. Shape (n, 2).
    :param values: the multivariate values associated with the data points. (n,)
    :param xi: the coordinates to evaluate the estimate at in 2 dimensions. Shape (m, 2).
    :param sigma: the gaussian kernel width (standard deviation). Float.
    :return: the coarse grained data at the coordinates xi. Shape (m,).
    """
    n: uint64 = points.shape[0]  # number of data points
    d: uint64 = 2  # dimension of the data points
    m: uint64 = xi.shape[0]  # number of evaluation points

    points_: float64[n, d] = points / sigma  # the scaled data points
    leaf_size: uint64 = np.floor(np.log2(n))  # the leaf_size of the KDTree (log2(n) is a good heuristic)
    tree = KDTree(points_, leafsize=leaf_size)  # the KDTree of the scaled data points

    xi_: float64[m, d] = xi / sigma  # the scaled evaluation points

    estimate: float64[m] = np.zeros(m)  # the estimate at the evaluation points
    norm: float64 = 1 / (2 * np.pi) / (sigma * sigma)  # the normalization factor of the gaussian kernel

    for j in prange(m):
        neighbors: uint64[:] = tree.query_radius(xi_[j, :], 10, workers=8)[0]
        for i in neighbors:
            estimate[j] += np.exp(-np.sum((points_[i, :] - xi_[j, :]) * (points_[i, :] - xi_[j, :])) / 2) \
                           * values[i]

    return estimate * norm


@jit(
    signature_or_function="float64[:,:](float64[:, :, :], float64[:, :], float64[:, :], float64)",
    nopython=True,
    cache=False,
    fastmath=False,
    parallel=False,
    debug=False,
    nogil=True,
    boundscheck=True
)
def coarse_grain_time_slices(points: float64[:, :, :],
                             values: float64[:, :],
                             xi: float64[:, :],
                             sigma: float64) -> float64[:, :]:
    """
    This function implements a gaussian coarse graining algorithm. Uses a KDTree to only consider nearby points.
    Heavily inspired by the scipy implementation at https://github.com/scipy/scipy/blob/main/scipy/stats/_stats.pyx
    :param points: the data points to estimate from in 2 dimensions + time. Shape (n, 2, t).
    :param values: the multivariate values associated with the data points. (n, t)
    :param xi: the coordinates to evaluate the estimate at in 2 dimensions. Shape (m, 2).
    :param sigma: the gaussian kernel width (standard deviation). Float.
    :return: the coarse grained data at the coordinates xi. Shape (m, t).
    """
    m: uint64 = xi.shape[0]  # number of evaluation points
    t: uint64 = points.shape[2]  # number of time slices

    estimate: float64[m, t] = np.zeros((m, t))  # the estimate at the evaluation points
    for h in range(t):
        estimate[:, h] = kd_gaussian_coarse_grain2d(points[:, :, h], values[:, h], xi[:, :], sigma)

    return estimate
