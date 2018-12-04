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
    # Repeat while cluster assignments don’t change
    print('Running k-means...')
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
    return cluster_assn, cluster_centroids

def run_kmeans(img, K):
    if len(img.shape) != 2:
        img = np.squeeze(img, axis=2)
    c_assn, c_cent = k_means(img, K)
    img_seg_output = np.zeros(shape=(c_assn.shape[0], c_assn.shape[1]))
    for i in range(img_seg_output.shape[0]):
        for j in range(img_seg_output.shape[1]):
            img_seg_output[i][j] = c_cent[int(c_assn[i][j])]
    img_seg_output = np.expand_dims(img_seg_output, axis=2)
    return img_seg_output





