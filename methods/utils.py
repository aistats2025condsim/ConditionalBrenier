import numpy as np

def scaled_euclidean_distance_squared(X, Y, beta_vect):
    """
    Compute the distance matrix between each pair of 
    vectors scaled by beta.
    Parameters:
        X    : {array}, shape (N1, dim)
        Y    : {array}, shape (N2, dim)
        beta : {vector}, shape (dim)
    Returns: 
        distances {array}, shape (N1, N2)
    """
    # check inputs
    assert(X.shape[1] == Y.shape[1])
    assert(X.shape[1] == len(beta_vect))
    # scale inputs
    X = X * np.sqrt(beta_vect)[np.newaxis, :]
    Y = Y * np.sqrt(beta_vect)[np.newaxis, :]
    XX = np.einsum('ij,ij->i', X, X)[:, np.newaxis]
    YY = np.einsum('ij,ij->i', Y, Y)[np.newaxis, :]
    # compute X^2 + Y^2 - 2XY
    distances = np.dot(X, Y.T)
    distances *= -2
    distances += XX
    distances += YY
    return distances
