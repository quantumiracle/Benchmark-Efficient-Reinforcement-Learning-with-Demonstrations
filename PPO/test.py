import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
from env import Reacher


mu=np.array([[-11.57634 ,  19.360188,  19.86773 ]])
sigma=np.array([[0.00635092, 0.0033142 , 0.00783274]])
sess = tf.Session()
d=tf.distributions.Normal(loc=mu, scale=sigma)

for i in range(30):
    print(sess.run(tf.squeeze(d.sample(1), axis=0)))