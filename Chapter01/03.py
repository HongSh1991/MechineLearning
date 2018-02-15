# -*-coding:utf8-*-
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

filename = sys.argv[1]
X = []
Y = []
with open(filename, 'r') as f:
	for line in f.readlines():
		xt, yt = [float(i) for i in line.split(',')]
		X.append(xt)
		Y.append(yt)

num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# 训练数据
X_train = np.array(X[:num_training]).reshape((num_training, 1))
Y_train = np.array(Y[:num_training])

# 测试数据
X_test = np.array(X[num_training:]).reshape((num_test, 1))
Y_test = np.array(Y[num_training:])

linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, Y_train)

Y_train_pred = linear_regressor.predict(X_train)
plt.figure()
plt.scatter(X_train, Y_train, color='green')
plt.plot(X_train, Y_train_pred, color='black', linewidth=4)
plt.title('Training data')
plt.show()
