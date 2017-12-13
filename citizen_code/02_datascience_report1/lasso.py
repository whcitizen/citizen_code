import numpy as np
import pandas as pd
from sklearn import linear_model

N = 10000
nfeat = 53

train_data = pd.read_csv("train.csv", header=None)
test_data = pd.read_csv("test.csv", header=None)

Train_X = train_data.iloc[:N, 0:nfeat].values
Test_X = train_data.iloc[N:, 0:nfeat].values
Train_Y = train_data.iloc[:N, 53].values.T
Test_Y = train_data.iloc[N:, 53].values.T

lasso = linear_model.Lasso(alpha=0)
lasso.fit(Train_X, Train_Y)

c = np.array(lasso.coef_)

yHatTrain = np.dot(Train_X, c)
for i in range(0, len(yHatTrain)):
    if yHatTrain[i] < 0:
        yHatTrain[i] = 0

yHatVal = np.dot(Test_X, c)
for i in range(0, len(yHatVal)):
    if yHatVal[i] < 0:
        yHatVal[i] = 0


print("Training error ", np.mean(np.abs(Train_Y - yHatTrain.T)))
print("Validation error ", np.mean(np.abs(Test_Y - yHatVal.T)))

"""
yHatTest = np.dot(np.array(test_data), c)
for i in range(0, len(yHatTest)):
    if yHatTest[i] < 0:
        yHatTest[i] = 0

np.savetxt('result_lasso_half95.txt', yHatTest)
"""