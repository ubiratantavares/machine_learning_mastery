import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 100)
y = np.linspace(-2, 2, 100)

# convert vector into 2D arrays
xx, yy = np.meshgrid(x,y)
# computation on matching
z = np.sqrt(1 - xx**2 - (yy/2)**2)

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
ax.set_zlim([0,2])
ax.plot_surface(xx, yy, z, cmap="cividis")
ax.view_init(45, 35)
plt.show()
