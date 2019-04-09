'''
this code is for supervised learning the state-action pair
using the neural network approach
'''
from __future__ import print_function
import tensorflow as tf

import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import argparse
import math
import time
import gzip
from env_2obstacle_10dim import Reacher
from a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch



save_file='./model/leakyrelu_small/ini'
# save_file='./model/ppo2'
data_file = open("data_memory2_21steps.p","rb")


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)


args = parser.parse_args()


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1, phase_train.name: True})
    
    error=tf.reduce_sum((abs(y_pre-v_ys)))

    
    result1 = sess.run(error, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})

    return result1

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,1,1], strides=[1,2,1,1], padding='SAME')

def leakyrelu(x, alpha=0.3, max_value=None):  #alpha need set
    '''ReLU.

    alpha: slope of negative section.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                             tf.cast(max_value, dtype=tf.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x

def full_batch_norm(x, n_out, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed



num_input=10
num_output=3
xs = tf.placeholder(tf.float32, [None, num_input])   
ys = tf.placeholder(tf.float32, [None, num_output])  
keep_prob = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)
phase_train = tf.placeholder(tf.bool, name='phase_train')

hidden_layer=400

#from common.models.py
def network_fn(X):
    num_layers=4
    num_hidden=100
    # activation=tf.tanh
    activation=tf.nn.relu
    layer_norm=False
    h = tf.layers.flatten(X)
    for i in range(num_layers):
        h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
        if layer_norm:
            h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
        h = activation(h)

    return h


with tf.variable_scope('actor'):
    x=network_fn(xs)
    x = tf.layers.dense(x, hidden_layer, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
    x=tf.nn.relu(x)

    x = tf.layers.dense(x, num_output, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

    # range of output -360-360 
    # prediction=30*tf.tanh(x)

    prediction = tf.nn.leaky_relu(x)


    # prediction=100*tf.tanh(x)
    # prediction = x


saver = tf.train.Saver()  #define saver of the check point

loss=tf.losses.mean_squared_error(ys, prediction)
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)


# rescale the action value to be in range 'scale', avoiding very large action (generated) on joint angles
def rescale_action(train_data, scale=None):
    #transformation angle range from inverse reacher (2PI) to original reacher (360)
    train_data=np.array(train_data)*180.0/np.pi
    max_action=np.max(np.abs(train_data))
    # print(max_action)
    if max_action>scale:
        return (scale/max_action)*(train_data)
    else:
        return (train_data)


loss_set=[]
step_set=[]
num_bat=1  # stack the batch to get larger training batch
test_bat=10 # size of test batch
screen_size=1000
# train_batches=50
# train_epoches=2000
train_epoches=100 
test_loss_set=[]

if args.train:
    # saver.restore(sess, save_file)

    test_set_s=[]
    test_set_a=[]
    state_set=[]
    action_set=[]
    try:
        # data_set = np.array(pickle.load(data_file))
        data_set=pickle.load(data_file)  # format:  [[s]], [[a]], [r], [[s']], [done] 
    except EOFError:
        print('Load data fails')
    for i in range(len(data_set)): # episodes: 50
        for j in range(len(data_set[i])): # steps: 21
            state_set.append(data_set[i][j][0][0])
            action_set.append(data_set[i][j][1][0])

    for i in range(test_bat):
        for j in range(len(data_set[i])):
            test_set_s.append(data_set[i][j][0][0])
            test_set_a.append(data_set[i][j][1][0])



    # for i in range (dev_bat+train_batches):
    #     print(i)
    #     try:
    #         # data_set = np.array(pickle.load(data_file))
    #         data_set=pickle.load(data_file)  # format:  [[s]], [[a]], [r], [[s']], [done] 
    #     except EOFError:
    #         break
    #     # print(data_set)
    #     row = len(data_set)
    #     print(row)
    #     if i>=dev_bat:

    #         for j in range (row):
    #             state_set.append(data_set[j][0][0])
    #             rescaled_action_set=rescale_action(data_set[j][1][0],360)
    #             action_set.append(rescaled_action_set)

            
    #     else:
    #         for j in range(row):
    #             test_set_s.append(data_set[j][0][0])
    #             test_set_a.append(rescale_action(data_set[j][1][0],360))

    data_file.close()



    for i in range(train_epoches):


        train_rep=1 # repeat training to make full use of each batch
        for j in range (train_rep):
            if i <0.3*train_epoches:
                pre,_, train_loss=sess.run([prediction,train_step,loss], feed_dict={xs: state_set, ys:action_set, \
                keep_prob: 0.9, lr:1e-3, phase_train.name: True})
                # print(pre)
            elif i< 0.6*train_epoches:
                pre,_, train_loss=sess.run([prediction,train_step,loss], feed_dict={xs: state_set, ys:action_set, \
                keep_prob: 0.9, lr:1e-3, phase_train.name: True})
            else:
                pre,_, train_loss=sess.run([prediction,train_step,loss], feed_dict={xs: state_set, ys:action_set, \
                keep_prob: 0.9, lr:3e-4, phase_train.name: True})
            print(action_set[300],pre[300])


        
        test_loss=sess.run(loss, feed_dict={xs: test_set_s, ys:test_set_a, \
            keep_prob: 0.9, lr:1e-5, phase_train.name: False})
        
        print(i, train_loss)
        # print(pre)
        loss_set.append(train_loss)
        test_loss_set.append(test_loss)
        step_set.append(i)
    # current loss
    plt.plot(step_set,loss_set,'b', label='Train Loss',alpha=0.7)
    # loss on a fixed test set
    plt.plot(step_set,test_loss_set,'r',label='Test Loss',alpha=0.7)

    plt.ylabel('Per Sample Loss')
    plt.xlabel('Epoch')
    plt.ylim(0,100)
    # set legend
    leg = plt.legend(loc=1)
    legfm = leg.get_frame()
    legfm.set_edgecolor('black') # set legend fame color
    legfm.set_linewidth(0.5) 
    saver.save(sess, save_file)
    # plt.ylim(0,2000)
    plt.savefig('train_curve.png')
    plt.show()


if args.test:


    # screen_size = 1000  # The size of the rendered screen in pixels
    link_lengths = [0.2, 0.14, 0.1]  # These lengths are in world space (0 to 1), not screen space (0 to 1000)
    # target_pose = [0.7, 0.7]
    range_pose=0.35
    joint_angles = [0.0,0.0,0.0]
    # joint_angles =np.array([0.1,0.1,0.1])*180.0/np.pi
    link_lengths=np.array(link_lengths)*screen_size
    # target_pose=np.array(target_pose)*screen_size
    target_pose=np.random.rand(2)*2*range_pose-range_pose
    target_screen = [int((0.5 + target_pose[0]) * screen_size), int((0.5 - target_pose[1]) * screen_size)]
    reacher=Reacher(screen_size, link_lengths, joint_angles)
    num_test_steps=50

    ini_action=np.array([0.1,0.1,0.1])
    pose=reacher.step([ini_action])
    saver.restore(sess, save_file)

    for i in range (num_test_steps):
        print(pose, target_screen)
        # state=np.concatenate((pose, target_screen))
        state = pose[0][0]
        print(state)

        action=sess.run(prediction, feed_dict={xs: [state], keep_prob: 1, lr:0.00001, phase_train.name: False})
        # action=np.array([[0.2,0.2,0.2]])
        print(action)
        pose=reacher.step(action)
        # print('jo: ', reacher.joint_angles)
        time.sleep(0.2)
        print(i,action[0])

    index_set=[i for i in range (num_test_steps)]

   
