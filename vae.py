import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from progress.bar import Bar

# get mnist data
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

inputs = tf.placeholder(tf.float32, [None, 784], name="inputs")

# hyperparameters
learning_rate = 0.001
encoding_dim = 2

with tf.name_scope("encoder"):
    fc1 = tf.layers.dense(inputs, 512, tf.nn.relu)
    # get mean and stddev
    mean = tf.layers.dense(fc1, encoding_dim, tf.nn.relu)
    stddev = tf.layers.dense(fc1, encoding_dim, tf.nn.relu)
    
dimensions = tf.shape(mean)
epsilon = tf.random.normal(dimensions, mean=0.0, stddev=1.0)
samples = mean + stddev * epsilon

with tf.name_scope("decoder"):
    fc2 = tf.layers.dense(samples, 512, tf.nn.relu)
    outputs = tf.layers.dense(fc2, 784, tf.nn.sigmoid)

with tf.name_scope("loss"):
    eps = 1e-8
    # loss is two parts-- first is the reconstruction loss
    reconstruction_loss = -tf.reduce_sum(inputs * tf.log(eps+outputs) + (1-inputs) * tf.log(eps+1-outputs), 1)
    # then we have the KL divergence between encoder gaussian and the prior
    # p(z) = N(z;0,1)
    kullback_leibner = 0.5 * tf.reduce_sum(-tf.log(eps+tf.square(stddev)) - 1 + tf.square(mean) + tf.square(stddev), 1)
    # total loss is reconstruction + KL
    loss = tf.reduce_mean(reconstruction_loss) + tf.reduce_mean(kullback_leibner)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

def train(sess, data, learning_rate, epochs=30, restore=False):
    sess.run(tf.global_variables_initializer())

    # we wanna save our spot so we don't have to constantly train this thing...
    saver = tf.train.Saver()
    
    if restore:
      saver.restore(sess, "./model/model.ckpt")
      print("restored!")

    for i in range(epochs):
        _loss, _ = sess.run([loss, optimizer],
            feed_dict={inputs:data})
        print("epoch %d, loss is %.03f" % (i, _loss))

        # save checkpoint every 5 epochs
        if i % 5 == 0:
            saver.save(sess, "./model/model.ckpt")
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
training_flag = False
if training_flag:
    train(sess, x_train, 0.01, epochs=10)

slice = x_test[:10]

n = slice.shape[0]  # how many digits we will display
plt.figure(figsize=(20, 4))
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, "model/model.ckpt")
compressed_imgs = sess.run(outputs, feed_dict={inputs:slice})
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(slice[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(compressed_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
