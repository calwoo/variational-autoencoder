import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

# get mnist data
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

inputs = tf.placeholder(tf.float32, [None, 28, 28, 1], name="inputs")

# hyperparameters
learning_rate = 0.01
encoding_dim = 10

def encoder(inputs):
    with tf.name_scope("encoder"):
        conv = tf.layers.conv2d(
            inputs=inputs,
            filters=1,
            kernel_size=[5,5],
            strides=[2,2],
            padding="valid",
            activation=tf.nn.relu)
        flat = tf.layers.flatten(conv)
        # get mean and stddev
        mean = tf.layers.dense(flat, encoding_dim, tf.nn.relu)
        std = tf.layers.dense(flat, encoding_dim, tf.nn.relu)
    return mean, std
    
def decoder(codes):
    with tf.name_scope("decoder"):
        fc_layer = tf.layers.dense(codes, 14*14*encoding_dim, tf.nn.relu)
        reshaped_codes = tf.reshape(fc_layer, [-1,14,14,encoding_dim])
        conv_t = tf.layers.conv2d_transpose(
            inputs=reshaped_codes,
            filters=1,
            kernel_size=[5,5],
            strides=[2,2],
            padding="same",
            activation=tf.nn.sigmoid)
    return conv_t

def samples(mean, std):
    dimensions = tf.shape(mean)
    epsilon = tf.random.normal(dimensions, mean=0.0, stddev=1.0)
    samples = mean + std * epsilon
    return samples

def autoencoder(input):
    mean, std = encoder(input)
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
        