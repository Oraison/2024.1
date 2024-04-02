from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

fig = plt.figure()
ax = plt.axes(projection="3d")

X = np.arange(0, 20, .2)
Y = np.arange(0, 20, .2)

X, Y = np.meshgrid(X, Y)

mu = [10, 10]
cov = [[1, 5], [1, 5]]
rv = sp.stats.multivariate_normal(mu, cov)
Z = rv.pdf(np.dstack([X, Y]))

ax.plot_surface(X, Y, Z, shade=False)

plt.show()