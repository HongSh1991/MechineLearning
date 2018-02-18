# -*-coding:utf8-*-
import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

data = np.array([[0.3, 0.2], [0.1, 0.4], [0.4, 0.6], [0.9, 0.5]])
labels = np.array([[0], [0], [0], [1]])

# define a preceptron
perceptron = nl.net.newp([[0, 1], [0, 1]], 1)
error = perceptron.train(data, labels, epochs=50, show=15, lr=0.01)

plt.figure()
"""plt.scatter(data[:, 0], data[:, 1])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Input data')"""
plt.plot(error)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.grid()
plt.title('Training error progress')

plt.show()
