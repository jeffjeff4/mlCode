import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=500, n_features=2, centers=3, random_state=23)

fig = plt.figure(0)
plt.grid(True)
plt.scatter(X[:,0], X[:,1])
plt.show()

k=3

clusters = {}
np.random.seed(23)

for idx in range(k):
    center = 2 * (2 * np.random.random((X.shape[1],)) - 1)
    points = []
    cluster = {
        'center': center,
        'points': []
    }

    clusters[idx] = cluster

clusters

plt.scatter(X[:, 0], X[:, 1])
plt.grid(True)
for idx in clusters:
    center = clusters[idx]['center']
    plt.scatter(center[0], center[1], marker='*', c='red')
plt.show()

def distance(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))

def assignClusters(X, clusters):
    for idx in range(X.shape[0]):
        dist = []
        curr_x = X[idx]

        for idx in range(k):
            dis0 = distance(curr_x, clusters[idx]['center'])
            dist.append(dis0)

        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)

    return clusters


def updateClusters(X, clusters):
    for idx in range(k):
        points = np.array(clusters[idx]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis=0)
            clusters[idx]['center'] = new_center
            clusters[idx]['points'] = []

    return clusters

def predCluster(X, clusters):
    pred = []
    for idx0 in range(X.shape[0]):
        dist = []
        for idx1 in range(k):
            tmp = distance(X[idx], clusters[idx1]['center'])
            dist.append(tmp)

        pred.append(np.argmin(dist))

    return pred

clusters = assignClusters(X, clusters)
clusters = updateClusters(X, clusters)
pred = predCluster(X, clusters)

plt.scatter(X[:,0], X[:, 1], c=pred)
for idx in clusters:
    center = clusters[idx]['center']
    plt.scatter(center[0], center[1], marker='^', c='red')
plt.show()