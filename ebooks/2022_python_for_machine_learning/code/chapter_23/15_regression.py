from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import numpy as np

# Generate 10-dimensional features and 1-dimensional targets
X, y = make_regression(n_samples=500, n_features=10, n_targets=1, n_informative=4,
                       noise=0.5, bias=-2.5, random_state=42)

# Run linear regression on the data
reg = LinearRegression()
reg.fit(X, y)

# Print the coefficient and intercept found
with np.printoptions(precision=5, linewidth=100, suppress=True):
    print(np.array(reg.coef_))
    print(reg.intercept_)
