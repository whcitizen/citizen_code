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

print("Training score ", len((np.where(yHatTrain == yTrain))[0])*1.0/XTrain.shape[0])
print("Validation score ", len((np.where(yHatVal == yVal))[0])*1.0/XVal.shape[0])

yHatTest = clf.predict(X_test)
np.savetxt('result_svm.txt', yHatTest)
