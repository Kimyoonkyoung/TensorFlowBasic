import tensorflow as tf

# operation
hello = tf.constant("hello, TensorFlow")

sess = tf.Session()

print sess.run(hello)