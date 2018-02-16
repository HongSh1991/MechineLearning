# -*-coding:utf8-*-
# import sys
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sklearn import linear_model

fileName = 'data_multi_variable.txt'
X = []
Y = []
with open(fileName, 'r') as f:
	for line in f.readlines():
		data = [float(i) for i in line.split(',')]
		xt, yt = data[:-1], data[-1]
		X.append(xt)
		Y.append(yt)

num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# 训练数据
X_train = np.array(X[:num_training])
Y_train = np.array(Y[:num_training])

# 测试数据
X_test = np.array(X[num_training:])
Y_test = np.array(Y[num_training:])

ridge_regressor = linear_model.Ridge(alpha=0.01, fit_intercept=True, max_iter=10000)
ridge_regressor.fit(X_train, Y_train)

output_model_file = 'saved_ridge_model.pkl'
with open(output_model_file, 'w') as f:
	pickle.dump(ridge_regressor, f)

with open(output_model_file, 'r') as f1:
	model_ridge = pickle.load(f1)

y_test_pred_ridge = model_ridge.predict(X_test)

plt.figure()
plt.scatter(X_test, Y_test, color='green')
plt.plot(X_test, y_test_pred_ridge, color='black', linewidth=4)
plt.show()
