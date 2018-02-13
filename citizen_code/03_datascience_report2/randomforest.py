import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as Classifier

nsample = 5000

X_train = np.array(pd.read_csv("X_train.csv",header=None))
y_train = list(pd.read_csv("y_train.csv",header=None).ix[:,0])
X_test = np.array(pd.read_csv("X_test.csv",header=None))
y_test = []

XTrain = X_train[:nsample,:] #use the first 4000 samples for training
yTrain = y_train[:nsample]
XVal = X_train[nsample:,:] #use the rests for validation
yVal = y_train[nsample:]

# mnist = fetch_mldata("MNIST original", data_home=".")
# data = np.asarray(mnist.data, np.float32)
# data_train, data_test, label_train, label_test = train_test_split(data, mnist.target, test_size=0.2)

classifier = Classifier()
classifier.fit(XTrain, yTrain)
test_prediction = classifier.predict(X_test)
train_prediction = classifier.predict(X_train)
val_prediction = classifier.predict(XVal)

valScore_val = 0
for i in range(len(yVal)):
  if val_prediction[i] == yVal[i]:
    valScore_val = valScore_val + 1

valScore_train = 0
for i in range(len(yTrain)):
  if train_prediction[i] == yTrain[i]:
    valScore_train = valScore_train + 1

print("Training score ", float(valScore_train)/len(yTrain))
print("Validation score ", float(valScore_val)/len(yVal))

np.savetxt("randomforest_importance.txt", classifier.feature_importances_)
np.savetxt("result_randomforest_nsample"+str(nsample) +".txt", test_prediction)