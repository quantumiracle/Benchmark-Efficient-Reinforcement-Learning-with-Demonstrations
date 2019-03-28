#!/usr/bin/env python3
import tensorflow as tf

x = tf.Variable(2.0)
y = x**2 + x - 1
# x = 2*y
# z = 3*x
p = 2*y
z = 3*p
grad = tf.gradients(z, x)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    grad_value = sess.run(grad)
    print(grad_value)



