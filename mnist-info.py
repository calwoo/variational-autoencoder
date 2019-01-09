import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

# get mnist data
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("x_train size: ", len(x_train))
print("x_test size: ", len(x_test))
print("y_train size: ", len(y_train))
print("y_test size: ", len(y_test))
print("image size: ", np.array(x_train[0]).shape)
