import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection
import torch

dataset = pd.read_csv("../data/automobile_data.csv", sep=r"\s*,\s*", header=0, engine="python")

# 1. Replace ? by NAN value
dataset = dataset.replace("?", np.nan)


# 2. Drop all column which has missing value - NAN value
dataset = dataset.dropna()
print(dataset.head())

# 1. symboling: -3, -2, -1, 0, 1, 2, 3.
# 2. normalized-losses: continuous from 65 to 256.
# 3. make:
# alfa-romero, audi, bmw, chevrolet, dodge, honda,
# isuzu, jaguar, mazda, mercedes-benz, mercury,
# mitsubishi, nissan, peugot, plymouth, porsche,
# renault, saab, subaru, toyota, volkswagen, volvo
#
# 4. fuel-type: diesel, gas.
# 5. aspiration: std, turbo.
# 6. num-of-doors: four, two.
# 7. body-style: hardtop, wagon, sedan, hatchback, convertible.
# 8. drive-wheels: 4wd, fwd, rwd.
# 9. engine-location: front, rear.
# 10. wheel-base: continuous from 86.6 120.9.
# 11. length: continuous from 141.1 to 208.1.
# 12. width: continuous from 60.3 to 72.3.
# 13. height: continuous from 47.8 to 59.8.
# 14. curb-weight: continuous from 1488 to 4066.
# 15. engine-type: dohc, dohcv, l, ohc, ohcf, ohcv, rotor.
# 16. num-of-cylinders: eight, five, four, six, three, twelve, two.
# 17. engine-size: continuous from 61 to 326.
# 18. fuel-system: 1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi.
# 19. bore: continuous from 2.54 to 3.94.
# 20. stroke: continuous from 2.07 to 4.17.
# 21. compression-ratio: continuous from 7 to 23.
# 22. horsepower: continuous from 48 to 288.
# 23. peak-rpm: continuous from 4150 to 6600.
# 24. city-mpg: continuous from 13 to 49.
# 25. highway-mpg: continuous from 16 to 54.
# 26. price: continuous from 5118 to 45400.

# 3. Selected features
col = ["make", "Feul-type", "Body-style", "horsepower"]
features = dataset[col]
labels = dataset[["price"]]

print(features.head())
print(labels.head())

# 4. Statistics for categorical values
pd.options.mode.chained_assignment = None       # Turn off warning

## Transform
print( features["horsepower"].describe() )
features["horsepower"] = pd.to_numeric(features["horsepower"])
print(features["horsepower"].describe())


print( labels["price"].describe() )
labels["price"] = labels.astype(float)
print( labels["price"].describe() )

### For string categorical value, use one-hot encoding for non-numeric values
features = pd.get_dummies(features, columns=["make", "Feul-type", "Body-style"])
print(features.head())
print(features.columns)

# 5. Standardize the numeric values
features[["horsepower"]] = preprocessing.scale(features[["horsepower"]])
print(features[["horsepower"]].head())


# 6. Split train and test dataset
X_train, x_test, Y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state=0)

# Convert to Torch Tensor
dtype = torch.float
X_train_tensor = torch.tensor(X_train.values, dtype=dtype)
x_test_tensor = torch.tensor(x_test.values, dtype=dtype)

Y_train_tensor = torch.tensor(Y_train.values, dtype=dtype)
y_test_tensor = torch.tensor(y_test.values, dtype=dtype)

print(X_train_tensor.shape)

# Init paras
input_size = 26
output_size = 1
hidden_size = 100
learning_rate = 0.0003
loss_fn = torch.nn.MSELoss()

# Structure network
model = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size), torch.nn.Sigmoid(), torch.nn.Linear(hidden_size, output_size))

# Run graph
for iter in range(10000):
    y_pred = model(X_train_tensor)
    loss = loss_fn(y_pred, Y_train_tensor)

    if iter % 1000 == 0:
        print("Iter: {}, loss: {}".format(iter, loss.item()))

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

# Take sample to test
sample = x_test.iloc[23]
print(sample)

sample_tensor = torch.tensor(sample.values, dtype=dtype)
y_pred = model(sample_tensor)
print("Predicted price of automobile is: ", int(y_pred.item()))
print("Actual price of automobile is: ", int(y_test.iloc[23]))

## Now test model by all test dataset
y_pred_tensor = model(x_test_tensor)
y_pred = y_pred_tensor.detach().numpy()

## Now draw to see
import matplotlib.pyplot as plt
plt.scatter(y_pred, y_test.values)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted prices vs Actual prices")
plt.show()


## Now save our model for the next time needed
torch.save(model, "my_model")
saved_model = torch.load("my_model")

y_pred_tensor = saved_model(x_test_tensor)
y_pred = y_pred_tensor.detach().numpy()

plt.figure(figsize=(15, 6))
plt.plot(y_pred, label="Predicted Price")
plt.plot(y_test.values, label="Actual Price")
plt.legend()
plt.show()