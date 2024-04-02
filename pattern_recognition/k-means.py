import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial import distance_matrix, distance
from random import sample

def k_means(X, num_of_cluster):
    centroid = X[sample(range(num_of_cluster), num_of_cluster)]
    prev_error = -1
    epsilon = 0.01
    covarage_count = 0
    
    while covarage_count <= 3:
        curr_error = 0
        cluster = [[] for _ in range(num_of_cluster)]

        for idx, dist in enumerate(distance.cdist(X, centroid, metric='euclidean')):
            min_idx = np.argmin(dist)
            cluster[min_idx].append(X[idx])
            curr_error += dist[min_idx]
        cluster = [np.array(x) for x in cluster]

        covarage_count = (covarage_count + 1) if prev_error < 0 or prev_error - curr_error > epsilon else 0
        
        centroid = np.array([np.mean(x, axis=0) for x in cluster])
        prev_error = curr_error
    
    return centroid, cluster
    
n_samples = 1000
# X, Y = make_blobs(n_samples=n_samples, random_state=42)
X, Y = make_blobs(n_samples=n_samples)

fig = plt.figure()
ax = plt.axes()

plt.xlim(-15, 15)
plt.ylim(-15, 15)

centroid, cluster = k_means(X, 7)

for c in cluster: ax.scatter(c[:,0], c[:,1])
ax.scatter(centroid[:,0], centroid[:,1], c='k')
plt.show()