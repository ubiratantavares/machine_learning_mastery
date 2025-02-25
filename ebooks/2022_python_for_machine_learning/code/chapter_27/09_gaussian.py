import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

mean = [0, 0]             # zero mean
cov = [[1, 0.8],[0.8, 1]] # covariance matrix
X1 = np.random.default_rng().multivariate_normal(mean, cov, 5000)
X2 = multivariate_normal.rvs(mean, cov, 5000)

fig = plt.figure(figsize=(12,6))
ax = plt.subplot(121)
ax.scatter(X1[:,0], X1[:,1], s=1)
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])
ax.set_title("NumPy")

ax = plt.subplot(122)
ax.scatter(X2[:,0], X2[:,1], s=1)
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])
ax.set_title("SciPy")

plt.show()
