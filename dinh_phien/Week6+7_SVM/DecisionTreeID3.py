import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TreeNode:
    def __init__(self, ids=None, children=[], entropy=0, depth=0):
        self.ids = ids                #index of data in this node
        self.children = children
        self.entropy = entropy
        self.depth = depth
        self.split_attribute = None  # which attribute is chosen, it non-leaf
        self.order = None            # order of values of split_attribute in children
        self.label = None            # label of node if it is a leaf

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        self.label = label


def entropy(fred):
    fred_0 = fred[np.array(fred).nonzero()[0]]
    prob_0 = fred_0 / float(fred_0.sum())
    entro = - np.sum(prob_0 * np.log(prob_0))
    return entro


class DecisionTreeID3:
    def __init__(self, max_depth=10, min_samples_split=2, min_gain=1e-4):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.Ntrain = 0

    def fit(self, data, target):
        self.Ntrain = data.count()[0]
        self.data = data
        self.attributes = list(data)
        self.target = target
        self.labels = target.unique()

        ids = range(self.Ntrain)
        self.root = TreeNode(ids=ids, entropy=self._entropy(ids), depth=0)
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.depth < self.max_depth or node.entropy <self.min_gain:
                if not node.label:
                    node.children = self._split(node)
                    if not node.children:
                        self._set_label(node)
                    queue += node.children
            else:
                self._set_label(node)

    def _entropy(self, ids):
                                                          # caculate entropy of a node with index ids
        if len(ids) == 0:
            return 0
        ids = [i +1 for i in ids]                         # panda series index starts from 1
        freq = np.array(self.target[ids].value_counts())
        return entropy(freq)

    def _set_label(self, node):
        # find label for a node if it is a leaf
        # simply chose by major voting
        target_ids = [i+1 for i in node.ids]                  # target is a series variable
        node.set_label(self.target[target_ids].mode()[0])     # most frequent label

    def _split(self, node):
        ids = node.ids
        best_gain = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.data.iloc[ids, :]
        for i, att in enumerate(self.attributes):
            values = self.data.iloc[: , i].unique().tolist()
            splits = []
            for val in values:
                sub_ids = sub_data.index[sub_data[att] == val].tolist()
                splits.append([sub_id-1 for sub_id in sub_ids])

            if max(map(len, splits)) == 1: continue   # gain = 0

            # don't split if a node has too small number of points
            if min(filter(lambda x : x >0, map(len, splits))) <  self.min_samples_split: continue

            # infomation gain:
            HxS = 0
            for split in splits:
                HxS += len(split) * self._entropy(split) /len(ids)
            gain = node.entropy - HxS
            if gain < self.min_gain: continue                    # stop if small gain
            if gain > best_gain:
                best_gain = gain
                best_splits = splits
                best_attribute = att
                order = values
        node.set_properties(best_attribute, order)
        child_nodes = []
        for split in best_splits:
            child = TreeNode(ids=split, entropy = self._entropy(split), depth= node.depth +1)
            if not split:
                index = [i +1 for i in node.ids]
                child.set_label(self.target[index].mode()[0])
            child_nodes.append(child)
        return child_nodes


    def predict(self, newdata):
        npoints = newdata.count()[0]
        labels = [None]*npoints
        for n in range(npoints):
            x = newdata.iloc[n, :]
            node = self.root
            while node.children:
                node = node.children[node.order.index(x[node.split_attribute])]
            labels[n] = node.label
        return labels


df = pd.DataFrame.from_csv('../Datasets/car.data2019.csv')
X_train = df.iloc[:1000, :-1]
y_train = df.iloc[:1000, -1]
X_test = df.iloc[1000:, :-1]
y_test = df.iloc[1000:, -1]

Tree = DecisionTreeID3(max_depth= 3, min_samples_split=2)
Tree.fit(X_train, y_train)
y_predict = Tree.predict(X_test)
# print(y_predict)

def evaluate(y, predict_y):

    count = 0
    for i in range(len(y)):
        if y[i] != predict_y[i]:
            count +=1
    return 100 * count / len(y)

error = evaluate(y_test.tolist(), y_predict)
print(error)
# Errors = []
# max_depths = list(range(2, 20))
#



# for i in max_depths:
#     Tree = DecisionTreeID3(max_depth= i, min_samples_split=2)
#     Tree.fit(X_train, y_train)
#     y_predict = Tree.predict(X_test)
#     eror = evaluate(y_test.tolist(), y_predict )
#     Errors.append(eror)

# Draw result:
# plt.plot(max_depths, Errors, "r-")
# plt.xlabel("Max_depths")
# plt.ylabel("Errors")
# plt.show()










