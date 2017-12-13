import numpy as np
import pandas as pd
import statsmodels.api as sm

N = 10000
nfeat = 53

train_data = pd.read_csv("train.csv", header=None)
test_data = pd.read_csv("test.csv", header=None)

Train_X = train_data.iloc[:N, 0:nfeat].values
Test_X = train_data.iloc[N:, 0:nfeat].values
Train_Y = train_data.iloc[:N, 53].values.T
Test_Y = train_data.iloc[N:, 53].values.T

x = sm.add_constant(Train_X)
y = Train_Y

model = sm.GLM(y, x, family=sm.families.Poisson())
results = model.fit()

w = pd.read_csv("glm_coef.csv", header=None)

yHatTrain = np.dot(Train_X, w)
for i in range(0, len(yHatTrain)):
    if yHatTrain[i] < 0:
        yHatTrain[i] = 0

yHatVal = np.dot(Test_X, w)
for i in range(0, len(yHatVal)):
    if yHatVal[i] < 0:
        yHatVal[i] = 0

print("Training error ", np.mean(np.abs(Train_Y - yHatTrain.T)))
print("Validation error ", np.mean(np.abs(Test_Y - yHatVal.T)))

yHatTest = np.dot(np.array(test_data),w)
for i in range(0, len(yHatTest)):
    if yHatTest[i] < 0:
        yHatTest[i] = 0

np.savetxt('result_GLM_half.txt', yHatTest)