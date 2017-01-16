import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

print 'x', x_data
print 'y', y_data

W = tf.Variable(tf.random_uniform([1,len(x_data)], -5.0, 5.0))

#hypothesis
hypothesis = tf.matmul(W, x_data)

#simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

#minimize
a = tf.Variable(0.1) #learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

#launch graph
sess = tf.Session()
sess.run(init)

#fit the line
for step in xrange(2001) :
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(cost), sess.run(W)