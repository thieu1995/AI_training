import matplotlib.pyplot as plt 
Y_pred = []
Y_true = []
with open("./result/flnn_res_2000_epoch_001.csv") as f:
    for line in f:
        temp = line.split(",")
        Y_pred.append(float(temp[0]))
        Y_true.append(float(temp[1]))
plt.plot(Y_true[:-1],color = 'b',label="actual data ")
plt.plot(Y_pred[:-1],color = 'r',label="predict")
RMSE = "RMSE:" + str(Y_pred[-1:])
MAE = "MAE" + str(Y_true[-1:])

plt.text(0,6,RMSE)
plt.text(0,5,MAE)
plt.legend(loc='best')
plt.show()