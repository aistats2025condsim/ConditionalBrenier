import numpy as np
#import torch
from sklearn import metrics
import ot

def empirical_wasserstein_distance(X, Y):
    """
    Compute the empirical Wasserstein distance between two sets of points.
    Parameters:
        X : {array}, shape (N1, dim)
        Y : {array}, shape (N2, dim)
    Returns:
        distance : {float}
    """
    # Compute the pairwise squared Euclidean distances
    M = ot.dist(X, Y, 'sqeuclidean')
    # Compute the optimal transport plan and the Wasserstein-2 distance 
    return ot.emd2([], [], M,numItermax=1000000)

def median_bandwidth_heuristic(X, Y):
    """
    Compute the median heuristic for the bandwidth of the Gaussian kernel.
    Parameters:
        X : {array}, shape (N1, dim)
        Y : {array}, shape (N2, dim)
    Returns:
        median : {float}
    """
    # check inputs
    assert(X.shape[1] == Y.shape[1])
    # compute distance matrix
    distances_X = metrics.pairwise_distances(X, metric='sqeuclidean')
    distances_Y = metrics.pairwise_distances(Y, metric='sqeuclidean')
    distances   = np.concatenate([distances_X.flatten(), distances_Y.flatten()])
    return np.sqrt(np.median(distances)/2.)

def mmd2(X, Y, kernel='rbf', sigma=None, degree=None, alpha=None, c=None):
    # compute kernel matrices
    if kernel == 'rbf':
        # compute bandwidth
        if sigma == None:
            sigma = median_bandwidth_heuristic(X, Y)
        K_XX, K_XY, K_YY = rbf_kernel(X, Y, sigma)
    elif kernel == 'poly':
        K_XX, K_XY, K_YY = poly_kernel(X, Y, d=degree, alpha=alpha, c=c)
    else:
        raise ValueError('Kernel is not implemented')
    return _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False)

def poly_kernel(X, Y, d, alpha, c):
    # check inputs
    assert(X.shape[1] == Y.shape[1])
    m = X.shape[0]
    # calculate the pairwise distance matrix
    Z = np.concatenate((X, Y), 0)
    ZZT = np.dot(Z,Z.T)
    # evaluate kernel
    K = (alpha * ZZT + c)**d
    # separate the kernel matrix
    return K[:m, :m], K[:m, m:], K[m:, m:]
    
def rbf_kernel(X, Y, sigma):
    # check inputs
    assert(X.shape[1] == Y.shape[1])
    m = X.shape[0]
    # calculate the pairwise distance matrix
    Z = np.concatenate((X, Y), 0)
    ZZT = np.dot(Z,Z.T)
    diag_ZZT = np.diag(ZZT).reshape(-1, 1)
    Z_norm_sqr = np.broadcast_to(diag_ZZT, ZZT.shape)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.T
    # evaluate kernel
    K = np.exp(-1. * exponent / (2. * sigma**2))
    # separate the kernel matrix
    return K[:m, :m], K[:m, m:], K[m:, m:]

################################################################################
# Helper functions to compute variances based on kernel matrices
################################################################################

def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False):
    m = K_XX.shape[0]  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = np.diag(K_XX)  # (m,)
        diag_Y = np.diag(K_YY)  # (m,)
        sum_diag_X = np.sum(diag_X)
        sum_diag_Y = np.sum(diag_Y)
        sum_diag2_X = np.dot(diag_X, diag_X)
        sum_diag2_Y = np.dot(diag_Y, diag_Y)

    Kt_XX_sums = np.sum(K_XX, axis=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = np.sum(K_YY, axis=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = np.sum(K_XY, axis=0)  # K_{XY}^T * e
    K_XY_sums_1 = np.sum(K_XY, axis=1)  # K_{XY} * e

    Kt_XX_sum = np.sum(Kt_XX_sums)  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = np.sum(Kt_YY_sums)  # e^T * \tilde{K}_YY * e
    K_XY_sum = np.sum(K_XY_sums_0)  # e^T * K_{XY} * e

    Kt_XX_2_sum = np.sum(K_XX ** 2) - sum_diag2_X  # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = np.sum(K_YY ** 2) - sum_diag2_Y  # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum = np.sum(K_XY ** 2)  # \| K_{XY} \|_F^2

    # evaluate the un-biased MMD^2 statistic
    mmd2 = Kt_XX_sum / (m * (m - 1)) + Kt_YY_sum / (m * (m - 1)) - 2.0 * K_XY_sum / (m * m)

    var_est = (
        2.0
        / (m**2 * (m - 1.0) ** 2)
        * (
            2 * Kt_XX_sums.dot(Kt_XX_sums)
            - Kt_XX_2_sum
            + 2 * Kt_YY_sums.dot(Kt_YY_sums)
            - Kt_YY_2_sum
        )
        - (4.0 * m - 6.0) / (m**3 * (m - 1.0) ** 3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4.0
        * (m - 2.0)
        / (m**3 * (m - 1.0) ** 2)
        * (K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
        - 4.0 * (m - 3.0) / (m**3 * (m - 1.0) ** 2) * (K_XY_2_sum)
        - (8 * m - 12) / (m**5 * (m - 1)) * K_XY_sum**2
        + 8.0
        / (m**3 * (m - 1.0))
        * (
            1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - np.dot(Kt_XX_sums, K_XY_sums_1)
            - np.dot(Kt_YY_sums, K_XY_sums_0)
        )
    )
    return mmd2, var_est

if __name__=='__main__':

    # define shapes and sizes
    Nx = 500
    Ny = 505
    dim = 2

    # Test empirical MMD distance with RBF
    X = np.random.randn(Nx, dim)
    Y = np.random.randn(Ny, dim)
    mmd2_est, mmd2_var = mmd2(X, Y, kernel='rbf')
    print('Empirical MMD^2 distance:', mmd2_est)
    print('Variance MMD^2 distance:', mmd2_var)

    # Test empirical MMD distance with poly kernel
    X = np.random.randn(Nx, dim)
    Y = np.random.randn(Ny, dim)
    mmd2_est, mmd2_var = mmd2(X, Y, kernel='poly', degree=2, alpha=1.0, c=2.0)
    print('Empirical MMD^2 distance:', mmd2_est)
    print('Variance MMD^2 distance:', mmd2_var)

