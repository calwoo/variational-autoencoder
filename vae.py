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
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# using these (28,28) images, we want to train a variational autoencoder. this is basically an autoencoder with
# stochastic units. this stochasticity will allow us to generate samples.

inputs = tf.placeholder(tf.float32, [None, 784], name="inputs")

def encoder(inputs):
    with tf.name_scope("encoder"):
        conv1 = tf.layers.conv2d(
            inputs=inputs,
            filters=16,
            kernel_size=[5,5],
            strides=[2,2],
            padding="valid",
            activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=32,
            kernel_size=[5,5],
            strides=[2,2],
            padding="valid",
            activation=tf.nn.relu)
        flat = tf.layers.flatten(conv2)
        # get mean and stddev
        codes = tf.layers.dense(flat, 32, tf.nn.relu)
    return codes
    
def decoder(codes):
    with tf.name_scope("decoder"):
        fc_layer = tf.layers.dense(codes, 7*7*32, tf.nn.relu)
        reshaped_codes = tf.reshape(fc_layer, [-1,7,7,32])
        conv2_t = tf.layers.conv2d_transpose(
            inputs=reshaped_codes,
            filters=16,
            kernel_size=[5,5],
            strides=[2,2],
            activation=tf.nn.relu)
        conv1_t = tf.layers.conv2d_transpose(
            inputs=conv2_t,
            filters=1,
            kernel_size=[5,5],
            strides=[2,2],
            activation=tf.nn.sigmoid)
        return conv1_t

def autoencoder(input):
    encoded = encoder(input)
    decoded = decoder(encoded)
    return decoded

def train(learning_rate, epochs=30):
    