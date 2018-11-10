import pandas as pd
from sklearn import preprocessing, model_selection

## Classified passenger will survise on titanic board or not

dataset = pd.read_csv("../data/titanic_train.csv")
print(dataset.head())

# 1. Drop un-wanted features
unwanted_features = ["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch", "Embarked"]
dataset = dataset.drop(unwanted_features, axis=1)
print(dataset.head())

# 2. Drop row which has missing value
dataset = dataset.dropna()

# 3. Preprocessing data
le = preprocessing.LabelEncoder()

dataset["Sex"] = le.fit_transform(dataset["Sex"])
print(dataset.head())

# 4. Split features and labels
features = ["Pclass", "Sex", "Age", "Fare"]
features = dataset[features]
print(features.head())

labels = dataset[["Survived"]]
print(labels.head())

# 5. Using one-hot encoder for field with decrete value
features = pd.get_dummies(features, columns=["Pclass"])
print(features.head())


# 6. Split train and test set
print("=========================================================================")
x_train, x_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state=0)
print(x_train.head())
print(y_train.head())

# 7. Using torch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

X_train = torch.from_numpy(x_train.values).float()
X_test = torch.from_numpy(x_test.values).float()

Y_train = torch.from_numpy(y_train.values).view(1, -1)[0]       # Single dimension [571]
Y_test = torch.from_numpy(y_test.values).view(1, -1)[0]

input_size = 6
output_size = 2
hidden_size = 10


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=-1)


model = Net()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.NLLLoss()

epoch_data = []
epochs = 1001

for epoch in range(1, epochs):
    optimizer.zero_grad()
    Y_pred = model(X_train)

    loss = loss_fn(Y_pred, Y_train)
    loss.backward()

    optimizer.step()

    # Calculate error on test, don't call backward(), we do not want to update weight when predict
    Y_pred_test = model(X_test)
    loss_test = loss_fn(Y_pred_test, Y_test)

    _, pred = Y_pred_test.data.max(1)       # Select the value with the highest probability - predicted value

    accuracy = pred.eq(Y_test.data).sum().item() / y_test.values.size
    epoch_data.append([epoch, loss.data.item(), loss_test.data.item(), accuracy])

    if epoch % 100 == 0:
        print("Epoch: %d (%d%%), Train loss : %.2f, Test loss : %.2f, Accuracy : %.3f" %(epoch, epoch/150 * 10, loss.data.item(), loss_test.data.item(), accuracy))



df_epochs_data = pd.DataFrame(epoch_data, columns=["epoch", "train_loss", "test_loss", "accuracy"])

import matplotlib.pyplot as plt
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
df_epochs_data[["train_loss", "test_loss"]].plot(ax=ax1)
df_epochs_data[["accuracy"]].plot(ax=ax2)
plt.ylim(ymin=0.5)
plt.show()

























