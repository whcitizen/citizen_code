from chainer import Chain, Variable, optimizers
import chainer.functions as F
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib
import numpy as np
import pandas as pd

# 学習データとテストデータに分ける
nsample = 4000

X_train = np.array(pd.read_csv("X_train.csv",header=None))
y_train = list(pd.read_csv("y_train.csv",header=None).ix[:,0])
X_test = np.array(pd.read_csv("X_test.csv",header=None))
y_test = []

XTrain = X_train[:nsample,:] #use the first 4000 samples for training
yTrain = y_train[:nsample]
XVal = X_train[nsample:,:] #use the rests for validation
yVal = y_train[nsample:]

# 層のパラメータ
model = Chain(
            conv1=F.Convolution2D(1, 32, 3),
            conv2=F.Convolution2D(32, 64, 3),
            l1=F.Linear(576, 200),
            l2=F.Linear(200, 100),
            l3=F.Linear(100, 10))

# 伝播のさせかた
def forward(x, is_train=True):
    h1 = F.max_pooling_2d(F.relu(model.conv1(x)), 3)
    h2 = F.max_pooling_2d(F.relu(model.conv2(h1)), 3)
    h3 = F.dropout(F.relu(model.l1(h2)), train=is_train)
    h4 = F.dropout(F.relu(model.l2(h3)), train=is_train)
    p = model.l3(h4)
    return p

# 学習のさせかた
optimizer = optimizers.Adam()
optimizer.setup(model)

# 学習
n_epoch = 100  # 学習繰り返し回数
batchsize = 20  # 学習データの分割サイズ
N = len(X_train)
losses = []  # 各回での誤差の変化を記録するための配列

for epoch in range(n_epoch):
    print('epoch: %d' % (epoch+1))
    perm = np.random.permutation(N)  # 分割をランダムにするための並べ替え
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N, batchsize):
        # 並べ替えた i〜i+batchsize 番目までのデータを使って学習
        x_batch = data_train[perm[i:i+batchsize]]
        t_batch = label_train[perm[i:i+batchsize]]

        # 初期化
        optimizer.zero_grads()

        # 順伝播
        x = Variable(x_batch)
        t = Variable(t_batch)
        p = forward(x)

        # 誤差の計算
        loss = F.softmax_cross_entropy(p, t)
        # 認識率の計算（表示用）
        accuracy = F.accuracy(p, t)

        # 誤差逆伝播
        loss.backward()

        # パラメータ更新
        optimizer.update()

        sum_loss += float(loss.data) * batchsize
        sum_accuracy += float(accuracy.data) * batchsize

    losses.append(sum_loss / N)
    print("loss: %f, accuracy: %f" % (sum_loss / N, sum_accuracy / N))

training_time = time.time() - start
joblib.dump((model, training_time, losses), "classifiers/"+"nn_cpu")


# 評価
start = time.time()
x_test = Variable(data_test)
result_scores = forward(x_test, is_train=False).data
predict_time = time.time() - start
results = np.argmax(result_scores, axis=1)

# %%
# 認識率を計算
score = accuracy_score(label_test, results)
print(training_time, predict_time)
print(score)
cmatrix = confusion_matrix(label_test, results)
print(cmatrix)
joblib.dump((training_time, predict_time, score, cmatrix), "results/"+"nn_cpu")