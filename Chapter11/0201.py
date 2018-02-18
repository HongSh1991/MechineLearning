# -*-coding:utf8-*-
import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

input_file = 'data_single_layer.txt'
input_text = np.loadtxt(input_file)
data = input_text[:, 0:2]
labels = input_text[:, 2:]

plt.figure()
plt.scatter(data[:, 0], data[:, 1])
plt.xlabel('X-label')
plt.ylabel('Y-label')
plt.title('Input data')
plt.show()

x_min, x_max = data[:, 0].min(), data[:, 0].max()
y_min, y_max = data[:, 1].min(), data[:, 1].max()
single_layer_net = nl.net.newp([[x_min, x_max], [y_min, y_max]], 2)

error = single_layer_net.train(data, labels, epochs=50, show=20, lr=0.01)

plt.figure()
plt.plot(error)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.grid()
plt.title('Training error progress')

plt.show()

print(single_layer_net.sim([[0.3, 4.5]]))
print(single_layer_net.sim([[4.5, 0.5]]))
print(single_layer_net.sim([[4.3, 8]]))
