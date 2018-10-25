
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.cluster import KMeans
# Load data:
data_set = load_boston().data
X = data_set[:, 4:6]
# Init k centroids in X:


def kmeans_init_centers(X, k):
    k_inits_index = np.random.choice(X.shape[0], k, replace="false")
    centroids = X[k_inits_index]
    return centroids

# Assign label:


def kmeans_asign_labels(X, centroids):

    sum_distances = 0
    Y_label = []       # each rows in X was assigned a label correspond with a element in Y_label

    for i in range(X.shape[0]):
        distance_centers = np.sqrt(np.sum((centroids - X[i]) ** 2, axis=1))
        sum_distances += np.min(distance_centers)
        center_index = np.argmin(distance_centers)
        Y_label.append(center_index)
    Y_label = np.asarray(Y_label)

    return Y_label, sum_distances


def kmeans_update_centers(X, K, Y_label):


    centroids = np.zeros((K, X.shape[1]))

    for k in range(K):

        Xk = X[Y_label == k]

        centroids[k] = np.mean(Xk, axis=0)
    return centroids


def k_means(X, k):
    centroids = kmeans_init_centers(X, k)
    array_sum_distances = [0]
    criterion = True
    while criterion:
        Y_label, sum_distances = kmeans_asign_labels(X, centroids)
        array_sum_distances.append(sum_distances)
        centroids = kmeans_update_centers(X, k, Y_label)
        if abs(array_sum_distances[-1] - array_sum_distances[-2]) < 1e-9:
            criterion = False

    return centroids, array_sum_distances[-1], Y_label


# implement k_means:

def display_result(X, K):

    centroids, sum_distances, Y_label = k_means(X, K)
    print(centroids)

    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    print('Centers found by scikit-learn:')
    print(kmeans.cluster_centers_)

    figure = plt.figure()
    sub_fig1 = figure.add_subplot(2, 1, 1)
    sub_fig1.scatter(X[:, 1], X[:, 0], c="r", marker="o")
    sub_fig2 = figure.add_subplot(2, 1, 2)
    colors = ["r", "b", "g", "k", "y", "m", "c"]
    markers = ["o", "^", "+", "x", ".", "*", "s"]
    for k in range(K):
        Xk = X[Y_label == k]
        i = k % 7
        j = k // 7
        # print(j)
        sub_fig2.scatter(Xk[:, 1], Xk[:, 0], c=colors[i], marker=markers[j])
    plt.show()


display_result(X, 7)






















