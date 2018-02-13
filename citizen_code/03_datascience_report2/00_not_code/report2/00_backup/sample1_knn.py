import numpy as np
import pandas as pd
import kNN as knn 

nsample = 4000

X_train = np.array(pd.read_csv("X_train.csv",header=None))
y_train = list(pd.read_csv("y_train.csv",header=None).ix[:,0])
X_test = np.array(pd.read_csv("X_test.csv",header=None))

XTrain = X_train[:nsample,:] #use the first 4000 samples for training
yTrain = y_train[:nsample]
XVal = X_train[nsample:,:] #use the rests for validation
yVal = y_train[nsample:]


#nVal = XVal.shape[0]
nVal = 100 #for simplicity...

valScore = 0
for i in range(nVal):
  prediction = knn.classify(XVal[i,:], XTrain, yTrain, 1) #1-NN
  print("Validation sample ", i, "...    Prediction: ", prediction, " Truth: ", yVal[i])
  if prediction == yVal[i]:
    valScore = valScore + 1

print("Validation score ", float(valScore)/nVal)

nTest = X_test.shape[0]
yHatTest = []
for i in range(nTest):
  prediction = knn.classify(X_test[i,:], XTrain, yTrain, 1) 
  print("Testing sample ", i, "...    Prediction: ", prediction)
  yHatTest.append(prediction)

np.savetxt('result_knn.txt', yHatTest)
