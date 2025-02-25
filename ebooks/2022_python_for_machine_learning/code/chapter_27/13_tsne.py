import datetime

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import numba

def tSNE(X, ndims=2, perplexity=30, seed=0, max_iter=500,
         stop_lying_iter=100, mom_switch_iter=400):
    """The t-SNE algorithm

	Args:
		X: the high-dimensional coordinates
		ndims: number of dimensions in output domain
    Returns:
        Points of X in low dimension
    """
    momentum = 0.5
    final_momentum = 0.8
    eta = 200.0
    N, _D = X.shape
    np.random.seed(seed)

    # normalize input
    X -= X.mean(axis=0) # zero mean
    X /= np.abs(X).max() # min-max scaled

    # compute input similarity for exact t-SNE
    P = computeGaussianPerplexity(X, perplexity)
    # symmetrize and normalize input similarities
    P = P + P.T
    P /= P.sum()
    # lie about the P-values
    P *= 12.0
    # initialize solution
    Y = np.random.randn(N, ndims) * 0.0001
    # perform main training loop
    gains = np.ones_like(Y)
    uY = np.zeros_like(Y)
    for i in range(max_iter):
        # compute gradient, update gains
        dY = computeExactGradient(P, Y)
        gains = np.where(np.sign(dY) != np.sign(uY), gains+0.2, gains*0.8).clip(0.1)
        # gradient update with momentum and gains
        uY = momentum * uY - eta * gains * dY
        Y = Y + uY
        # make the solution zero-mean
        Y -= Y.mean(axis=0)
        # Stop lying about the P-values after a while, and switch momentum
        if i == stop_lying_iter:
            P /= 12.0
        if i == mom_switch_iter:
            momentum = final_momentum
        # print progress
        if (i % 50) == 0:
            C = evaluateError(P, Y)
            now = datetime.datetime.now()
            print(f"{now} - Iteration {i}: Error = {C}")
    return Y

@numba.jit(nopython=True)
def computeExactGradient(P, Y):
    """Gradient of t-SNE cost function

	Args:
        P: similarity matrix
        Y: low-dimensional coordinates
    Returns:
        dY, a numpy array of shape (N,D)
	"""
    N, _D = Y.shape
    # compute squared Euclidean distance matrix of Y, the Q matrix, and the
    # normalization sum
    DD = computeSquaredEuclideanDistance(Y)
    Q = 1/(1+DD)
    sum_Q = Q.sum()
    # compute gradient
    mult = (P - (Q/sum_Q)) * Q
    dY = np.zeros_like(Y)
    for n in range(N):
        for m in range(N):
            if n==m: continue
            dY[n] += (Y[n] - Y[m]) * mult[n,m]
    return dY

@numba.jit(nopython=True)
def evaluateError(P, Y):
    """Evaluate t-SNE cost function

    Args:
        P: similarity matrix
        Y: low-dimensional coordinates
    Returns:
        Total t-SNE error C
    """
    DD = computeSquaredEuclideanDistance(Y)
    # Compute Q-matrix and normalization sum
    Q = 1/(1+DD)
    np.fill_diagonal(Q, np.finfo(np.float32).eps)
    Q /= Q.sum()
    # Sum t-SNE error: sum P log(P/Q)
    error = P * np.log( (P + np.finfo(np.float32).eps)
                       / (Q + np.finfo(np.float32).eps) )
    return error.sum()

@numba.jit(nopython=True)
def computeGaussianPerplexity(X, perplexity):
    """Compute Gaussian Perplexity

    Args:
        X: numpy array of shape (N,D)
        perplexity: double
    Returns:
        Similarity matrix P
    """
    # Compute the squared Euclidean distance matrix
    N, _D = X.shape
    DD = computeSquaredEuclideanDistance(X)
    # Compute the Gaussian kernel row by row
    P = np.zeros_like(DD)
    for n in range(N):
        found = False
        beta = 1.0
        min_beta = -np.inf
        max_beta = np.inf
        tol = 1e-5

        # iterate until we get a good perplexity
        n_iter = 0
        while not found and n_iter < 200:
            # compute Gaussian kernel row
            P[n] = np.exp(-beta * DD[n])
            P[n,n] = np.finfo(np.float32).eps
            # compute entropy of current row
            # Gaussians to be row-normalized to make it a probability
            # then H = sum_i -P[i] log(P[i])
            #        = sum_i -P[i] (-beta * DD[n] - log(sum_P))
            #        = sum_i P[i] * beta * DD[n] + log(sum_P)
            sum_P = P[n].sum()
            H = beta * (DD[n] @ P[n]) / sum_P + np.log(sum_P)
            # Evaluate if entropy within tolerance level
            Hdiff = H - np.log2(perplexity)
            if -tol < Hdiff < tol:
                found = True
                break
            if Hdiff > 0:
                min_beta = beta
                if max_beta in (np.inf, -np.inf):
                    beta *= 2
                else:
                    beta = (beta + max_beta) / 2
            else:
                max_beta = beta
                if min_beta in (np.inf, -np.inf):
                    beta /= 2
                else:
                    beta = (beta + min_beta) / 2
            n_iter += 1
        # normalize this row
        P[n] /= P[n].sum()
    assert not np.isnan(P).any()
    return P

@numba.jit(nopython=True)
def computeSquaredEuclideanDistance(X):
    """Compute squared distance
    Args:
        X: numpy array of shape (N,D)
    Returns:
        numpy array of shape (N,N) of squared distances
    """
    N, _D = X.shape
    DD = np.zeros((N,N))
    for i in range(N-1):
        for j in range(i+1, N):
            diff = X[i] - X[j]
            DD[j][i] = DD[i][j] = diff @ diff
    return DD

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# pick 1000 samples from the dataset
rows = np.random.choice(X_test.shape[0], 1000, replace=False)
X_data = X_train[rows].reshape(1000, -1).astype("float")
X_label = y_train[rows]
# run t-SNE to transform into 2D and visualize in scatter plot
Y = tSNE(X_data, 2, 30, 0, 500, 100, 400)
plt.figure(figsize=(8,8))
plt.scatter(Y[:,0], Y[:,1], c=X_label)
plt.show()
