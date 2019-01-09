"""
I wanted to build a variational autoencoder for MNIST just for see that I could. There will probably be an
adjoining ipython notebook to describe the math more throughly.
"""

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

# get mnist data
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(len(x_train))
plt.imshow(x_train[0])
plt.show()