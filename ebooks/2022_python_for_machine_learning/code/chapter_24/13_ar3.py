import numpy as np
import matplotlib.pyplot as plt

# Predefined paramters
ar_n = 3                     # Order of the AR(n) data
ar_coeff = [0.7, -0.3, -0.1] # Coefficients b_3, b_2, b_1
noise_level = 0.1            # Noise added to the AR(n) data
length = 200                 # Number of data points to generate

# Random initial values
ar_data = list(np.random.randn(ar_n))

# Generate the rest of the values
for i in range(length - ar_n):
    next_val = (ar_coeff @ np.array(ar_data[-3:])) + np.random.randn() * noise_level
    ar_data.append(next_val)

# Plot the time series
fig = plt.figure(figsize=(12,5))
plt.plot(ar_data)
plt.show()
