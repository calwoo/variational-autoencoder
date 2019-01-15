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
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

# using these (28,28) images, we want to train a variational autoencoder. this is basically an autoencoder with
# stochastic units. this stochasticity will allow us to generate samples.

inputs = tf.placeholder(tf.float32, [None, 28, 28, 1], name="inputs")

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
        codes = tf.layers.dense(flat, 10, tf.nn.relu)
    return codes
    
def decoder(codes):
    with tf.name_scope("decoder"):
        fc_layer = tf.layers.dense(codes, 7*7*10, tf.nn.relu)
        reshaped_codes = tf.reshape(fc_layer, [-1,7,7,10])
        conv2_t = tf.layers.conv2d_transpose(
            inputs=reshaped_codes,
            filters=16,
            kernel_size=[5,5],
            strides=[2,2],
            padding="same",
            activation=tf.nn.relu)
        conv1_t = tf.layers.conv2d_transpose(
            inputs=conv2_t,
            filters=1,
            kernel_size=[5,5],
            strides=[2,2],
            padding="same",
            activation=tf.nn.sigmoid)
    return conv1_t

def autoencoder(input):
    encoded = encoder(input)
    decoded = decoder(encoded)
    return decoded

def loss_function(input, learning_rate):
    outputs = autoencoder(input)
    loss = -tf.reduce_mean(input * tf.log(outputs) + (1-input) * tf.log(1-outputs))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return loss, optimizer

def train(data, learning_rate, epochs=30):
    with tf.Session() as sess:
        loss, optimizer = loss_function(inputs, learning_rate)
        sess.run(tf.global_variables_initializer())

        for i in range(epochs):
            _loss, _ = sess.run([loss, optimizer],
                feed_dict={inputs:data})
            print("epoch %d, loss is %.03f" % (i, _loss))

train(x_train, 0.01)

"""with tf.Session() as sess:
    ae = autoencoder(inputs)
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.shape(inputs), feed_dict={inputs:x_train}))
    print(sess.run(tf.shape(ae), feed_dict={inputs:x_train}))"""

        