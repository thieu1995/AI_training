import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split


# preprocess data:
def pre_process(X):
    X_max = np.max(X, axis=0)
    return X / X_max

# Load dataset:
def load_dataset(datasetX, datasetY, size_test):
    X_train, X_test, Y_train, Y_test = train_test_split(datasetX, datasetY, test_size=size_test, random_state=5)
    X_train = pre_process(X_train)
    X_test = pre_process(X_test)
    Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
    Y_test = np.reshape(Y_test, (Y_test.shape[0], 1))
    return X_train, X_test, Y_train, Y_test

def Find_k_nearest_neighbors(u, X_train, Y_train, K, Weights):

    distances = np.sqrt(np.dot((X_train - u)**2, Weights))
    indices_sort_distances = np.argsort(distances, axis=0)
    mask_k_nearest_neighbors = ma.masked_outside(indices_sort_distances, K, float("Inf")).mask
    distances_k_nearest_neighbors = distances[mask_k_nearest_neighbors]
    Label_k_nearest_neighbors = Y_train[mask_k_nearest_neighbors]
    return distances_k_nearest_neighbors, Label_k_nearest_neighbors


def Assign_label(distance_k_nearest_neighbors, Y_k_nearest_neighbors, anpha):
    Y_range = np.unique(Y_k_nearest_neighbors, axis=0)
    affections = []
    for i in range(Y_range.shape[0]):
        label_i = Y_range[i]
        distances_label_i = distance_k_nearest_neighbors[Y_k_nearest_neighbors == label_i]
        affection = np.sum(1 / (distances_label_i + anpha * np.ones(distances_label_i.shape)), axis=0)
        affections.append(affection)
    index_label = np.argmax(affections)
    return Y_range[index_label]


# Evalueate model:


def Evaluate(datasetX, datasetY, Weights, K):
    X_train, X_test, Y_train, Y_test = load_dataset(datasetX, datasetY, 0.33)
    assign_label = []
    for i in range(X_test.shape[0]):
        distance_k_nearest_neighbors, Label_k_nearest_neighbors = Find_k_nearest_neighbors(X_test[i], X_train, Y_train, K, Weights)
        labeli = Assign_label(distance_k_nearest_neighbors, Label_k_nearest_neighbors, 1e-9)
        assign_label.append(labeli)
    check_array = Y_test - np.array(assign_label).reshape((Y_test.shape[0]), 1)
    error_percent = 100 * np.count_nonzero(check_array, axis=0) / (Y_test.shape[0])
    return error_percent

# Use sklearn:
def Use_sklearn(datasetX, datasetY, k, size_test):
    X_train, X_test, Y_train, Y_test = train_test_split(datasetX, datasetY, test_size=size_test, random_state=5)
    clf = neighbors.KNeighborsClassifier(n_neighbors=k, p=2)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    check_array = Y_test - Y_pred
    error_percent = 100 * np.count_nonzero(check_array, axis=0) / (Y_test.shape[0])
    return error_percent

# Display results:


Weights = np.array([[0.25, 0.25, 0.25, 0.25]]).T
K = 8
iris = datasets.load_iris()
datasetX = iris.data
datasetY = iris.target
M = 20
errors_percent = []
error_sklearns = []
for k in range(1, M +1):
    error = Evaluate(datasetX, datasetY, Weights, k)
    error_skl = Use_sklearn(datasetX, datasetY, k, 0.33)
    errors_percent.append(error)
    error_sklearns.append(error_skl)
k_array = np.arange(1, M + 1)
# print(errors)

plt.plot(k_array, errors_percent, "b-")
plt.plot(k_array, error_sklearns, "r-")
plt.ylabel('Errors %')
plt.xlabel('Values of K')
plt.show()











