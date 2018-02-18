# -*-coding:utf8-*-
import neurolab as nl
import numpy as np
import matplotlib.pyplot as plt

# 生成训练数据
min_value = -12
max_value = 12
num_datapoints = 90

x = np.linspace(min_value, max_value, num_datapoints)
y = 2 * np.square(x) + 7
y /= np.linalg.norm(y)

# reshape
data = x.reshape(num_datapoints, 1)
labels = y.reshape(num_datapoints, 1)

plt.figure()
plt.ylim(-0.05, 0.25)
plt.xlim(-15, 15)
plt.scatter(data, labels)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Input data')
plt.show()

multilayer_net = nl.net.newff([[min_value, max_value]], [10, 10, 1])
multilayer_net.trainf = nl.net.train.train_gd
error = multilayer_net.train(data, labels, epochs=800, show=100, goal=0.01)
pred_output = multilayer_net.sim(data)

plt.figure()
plt.xlim(0, 800)
plt.ylim(0, 35)
plt.plot(error)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')
plt.show()

x2 = np.linspace(min_value, max_value, num_datapoints * 2)
y2 = multilayer_net.sim(x2.reshape(x2.size, 1)).reshape(x2.size)
y3 = pred_output.reshape(num_datapoints)

plt.figure()
plt.ylim(0.00, 0.25)
plt.xlim(-15, 15)
plt.plot(x2, y2, '-', x, y, '.', x, y3, 'p')
plt.title('Ground truth vs predicted output')
plt.show()
