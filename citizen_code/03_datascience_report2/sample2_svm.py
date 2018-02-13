import numpy as np
import pandas as pd
from sklearn import svm

nsample = 4000

X_train = np.array(pd.read_csv("X_train.csv",header=None))
y_train = np.array(pd.read_csv("y_train.csv",header=None).ix[:,0])
X_test = np.array(pd.read_csv("X_test.csv",header=None))

XTrain = X_train[:nsample,:] #use the first 4000 samples for training
yTrain = y_train[:nsample]
XVal = X_train[nsample:,:] #use the rests for validation
yVal = y_train[nsample:]

print("Training linear SVM classifier.")
clf = svm.LinearSVC()
clf.fit(XTrain,yTrain)
yHatTrain = clf.predict(XTrain)
yHatVal = clf.predict(XVal)

nVal = 100 #for simplicity...

valScore_val = 0
for i in range(nVal):
  if yHatVal[i] == yVal[i]:
    valScore_val = valScore_val + 1

valScore_train = 0
for i in range(nVal):
  if yHatTrain[i] == yTrain[i]:
    valScore_train = valScore_train + 1

print("Training score ", float(valScore_train)/nVal)
print("Validation score ", float(valScore_val)/nVal)

"""
yHatTest = clf.predict(X_test)
np.savetxt('result_svm.txt', yHatTest)
"""