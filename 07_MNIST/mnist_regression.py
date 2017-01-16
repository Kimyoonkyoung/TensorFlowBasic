from random import randint

import tensorflow as tf
import numpy as np
import os
import urllib
import gzip

from tensorflow.examples.tutorials.mnist import mnist

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

def maybe_download(filename, work_directory):
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath,_ = urllib.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)

def extract_images(filename):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file : %s' % (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

batch_size = 5
training_epochs = 10
display_step = 100
learning_rate = 0.01

#softmax
activation = tf.nn.softmax(tf.matmul(x, W)+b)

#minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#training cycle
sess = tf.Session()
for epoch in range(training_epochs) :
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples / batch_size)

    #loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys})/total_batch

    if epoch % display_step == 0:
        print "Epoch : ", '%04d' % (epoch + 1), "cost = ", "{:.9f}".format(avg_cost)

print "Optimization Finisihed!"

r = randint(0, mnist.test.num_examples -1)
print "Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1],1))
print "Prediction: ", sess.run(tf.argmax(activation,1), {x:mnist.test.images[r:r+1]})

#plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
#plt.show()