import numpy as np

def k_means(input, K):
    x_dim = input.shape[0]
    y_dim = input.shape[1]
    cluster_centroids = []
    c_changed = True
    counter = 0
    cluster_assn = np.zeros(shape=(x_dim, y_dim))
    # initialize cluster centroids
    cluster_centroids = list(np.linspace(np.min(input), np.max(input), K))
    print(cluster_centroids)
    # Repeat while cluster assignments don’t change
    while c_changed:
        c_changed = False
        counter += 1
        print(counter)
        # Assign each pixel to the nearest centroid using Euclidean distance
        for i in range(x_dim):
            for j in range(y_dim):
                c_i = []
                for mu in cluster_centroids:
                    c_i.append(np.abs(input[i][j] - mu))
                cluster_assn[i][j] = np.argmin(c_i)
        # Given new assignments, compute new cluster centroids as mean of all points in cluster
        for k in range(K):
            cluster_k = input[cluster_assn == k]
            centroid_k = cluster_k.mean()
            if not centroid_k == cluster_centroids[k]:
                c_changed = True
                cluster_centroids[k] = centroid_k
        print(cluster_centroids)
    return cluster_assn, cluster_centroids






