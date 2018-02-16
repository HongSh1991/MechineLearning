# -*-coding:utf8-*-
# import sys
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sklearn import linear_model

filename = "data_singlevar.txt"
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

# 保存训练数据训练的模型
output_model_file = 'save_model.pkl'
with open(output_model_file, 'w') as f:
	pickle.dump(linear_regressor, f)

with open(output_model_file, 'r') as f1:
	model_linregr = pickle.load(f1)

y_test_pred_new = model_linregr.predict(X_test)
print("\nNew mean absolute error = ", round(sm.mean_absolute_error(Y_test, y_test_pred_new), 2))

"""Y_train_pred = linear_regressor.predict(X_train)
plt.figure()
plt.scatter(X_train, Y_train, color='green')
plt.plot(X_train, Y_train_pred, color='black', linewidth=4)
plt.title('Training data')"""

y_test_pred = linear_regressor.predict(X_test)

plt.scatter(X_test, Y_test, color='green')
plt.plot(X_test, y_test_pred_new, color='black', linewidth=4)
plt.title('Test data')
plt.show()
