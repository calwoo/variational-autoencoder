import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from progress.bar import Bar

# get mnist data
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

inputs = tf.placeholder(tf.float32, [None, 28, 28, 1], name="inputs")

# hyperparameters
learning_rate = 0.01
encoding_dim = 2

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

def vae(input):
    mean, std = encoder(input)
    codes = samples(mean, std)
    decoded = decoder(codes)
    return mean, std, decoded

def loss_function(input, learning_rate):
    o_means, o_std, outputs = vae(input)
    eps = 1e-6
    # loss is two parts-- first is the reconstruction loss
    reconstruction_loss = -tf.reduce_sum(input * tf.log(eps+outputs) + (1-input) * tf.log(eps+1-outputs), 1)
    # then we have the KL divergence between encoder gaussian and the prior
    # p(z) = N(z;0,1)
    kullback_leibner = 0.5 * tf.reduce_sum(tf.log(eps+tf.square(o_std)) - 1 + tf.square(o_means) + tf.square(o_std), 1)
    # total loss is reconstruction + KL
    loss = tf.reduce_mean(reconstruction_loss) + tf.reduce_mean(kullback_leibner)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return loss, optimizer
    

def train(sess, data, learning_rate, epochs=30):
    loss, optimizer = loss_function(inputs, learning_rate)
    sess.run(tf.global_variables_initializer())

    # we wanna save our spot so we don't have to constantly train this thing...
    saver = tf.train.Saver()

    for i in range(epochs):
        _loss, _ = sess.run([loss, optimizer],
            feed_dict={inputs:data})
        print("epoch %d, loss is %.03f" % (i, _loss))

        # save checkpoint every 5 epochs
        if i % 5 == 0:
            saver.save(sess, "./model/model")
            print("saved the model for ya, chief!")

def test():
    # display a 2D manifold of the digits
    n = 10  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # we will sample n points within [-15, 15] standard deviations
    grid_x = np.linspace(-15, 15, n)
    grid_y = np.linspace(-15, 15, n)
    # progress bar
    bar = Bar("creating image...")

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            # tensor inputs
            code_input = tf.placeholder(tf.float32, [1,2])
            x_decoder = decoder(code_input)
            # set up session to run decoder
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                x_decoded = sess.run([x_decoder], feed_dict={code_input: z_sample})
            # update progress bar
            bar.next()
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit
    bar.finish()
    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.show()

sess = tf.Session()
training_flag = True
if training_flag:
    train(sess, x_train, 0.01, epochs=50)
test()

# testing ground
with tf.Session() as sess:
    data = x_train[:2].reshape(2, *x_train[0].shape)
    mean, std, decoded = vae(inputs)
    loss, _ = loss_function(inputs, learning_rate)
    sess.run(tf.global_variables_initializer())
    m, s, d = sess.run([mean, std, decoded], feed_dict={inputs:data})
    _loss = sess.run([loss], feed_dict={inputs:x_train})
    print("mean: ", m)
    print("stddev: ", s)
    # print("decoded: ", d)
    print("loss: ", _loss)