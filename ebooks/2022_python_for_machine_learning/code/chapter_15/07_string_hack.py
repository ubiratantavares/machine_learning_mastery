import numpy as np
import scipy.stats as stats

# Covariance matrix and Cholesky decomposition
cov = np.array([[1, 0.8], [0.8, 1]])
L = np.linalg.cholesky(cov)

# Generate 100 pairs of bi-variate Gaussian random numbers
if not "USE SCIPY":
   z = np.random.randn(100,2)
   x = z @ L.T
else:
   x = stats.multivariate_normal(mean=[0, 0], cov=cov).rvs(100)
