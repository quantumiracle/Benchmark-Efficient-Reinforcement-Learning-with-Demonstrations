"""
A simple version of Proximal Policy Optimization (PPO) using single thread.
Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]
View more on my tutorial website: https://morvanzhou.github.io/tutorials
Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
from env import Reacher
import time
# np.set_printoptions(precision=4, threshold=np.nan)


EP_MAX = 2000
EP_LEN = 10
GAMMA = 0.9
A_LR = 1e-4
C_LR = 2e-4
# A_LR = 1e-10
# C_LR = 2e-10
BATCH = 64
A_UPDATE_STEPS = 3
C_UPDATE_STEPS = 2
S_DIM, A_DIM = 10,3
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization


class PPO(object):

    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        self.sigma_lower_bound=1e-3

        # critic
        with tf.variable_scope('critic',reuse=tf.AUTO_REUSE):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        # initial policy
        pi, pi_params = self._build_anet_ini('pi', trainable=False, sigma_lower_bound=self.sigma_lower_bound)
        # residual policy
        res_pi, res_pi_params = self._build_anet_res('res_pi', trainable=True, sigma_lower_bound=self.sigma_lower_bound)
        oldpi, oldpi_params = self._build_anet_res('oldpi', trainable=False, sigma_lower_bound=self.sigma_lower_bound)
        # initial
        with tf.variable_scope('sample_action',reuse=tf.AUTO_REUSE):
            self.sample_ini_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        # ini_action= tf.constant(self.sample_ini_op)
        # residual
        with tf.variable_scope('sample_action',reuse=tf.AUTO_REUSE):
            self.sample_res_op = tf.squeeze(res_pi.sample(1), axis=0) 

        trade_off=0.00000001
        self.sample_op=trade_off*self.sample_ini_op+self.sample_res_op
        # self.sample_op=tf.reduce_sum([self.sample_ini_op, self.sample_res_op], axis=0)
        ''' non-stable gradients of tf, some small value causes inf gradients -> nan in weights, even plus a constant here cannot work'''
        # self.sample_op=self.sample_res_op 
        # self.sample_op=tf.math.add(self.sample_ini_op, self.sample_res_op)
        with tf.variable_scope('update_oldpi', reuse=tf.AUTO_REUSE):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(res_pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = res_pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, res_pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.log(tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv)))
            # add entropy to boost exploration
            # entropy=pi.entropy()
            # self.aloss-=0.1*entropy
        with tf.variable_scope('atrain', reuse=tf.AUTO_REUSE):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)


        self.sess.run(tf.global_variables_initializer())
        ''' initialize the actor policy, both pi and oldpi'''
        self.load_ini()
        self.sess.run(self.update_oldpi_op)

        actor_var_list=tf.contrib.framework.get_variables('res_pi/dense/bias:0')
        print(self.sess.run(actor_var_list))


        # check if loaded correctly
        # actor_var_list=tf.contrib.framework.get_variables('pi')
        # print(self.sess.run(actor_var_list))
        


    # def load_ini(self, ):
    #     actor_var_list=tf.contrib.framework.get_variables('pi')
    #     print(actor_var_list)
    #     actor_saver=tf.train.Saver(actor_var_list)
    #     actor_saver.restore(self.sess, './ini/ppo_fixed')
    #     print('Actor Load Succeed!')


    def load_ini(self, ):
        variables = tf.contrib.framework.get_variables_to_restore()
        # dense_5 is layer of sigma, sigma is fixed for pretrained
        actor_var_list = [v for v in variables if (v.name.split('/')[0]=='pi' and v.name.split('/')[1]!='dense_5')]
        # print(actor_var_list)
        actor_saver=tf.train.Saver(actor_var_list)
        actor_saver.restore(self.sess, './ini/ppo_fixed')
        print('Actor Load Succeed!')

    def pretrain_critic(self,s,a, r):
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]



    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for i in range(A_UPDATE_STEPS):
                print('updata: ',i)
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            # [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]
            for _ in range(A_UPDATE_STEPS):
                # print(s,a,adv)
                loss,_= self.sess.run([self.aloss, self.atrain_op], {self.tfs: s, self.tfa: a, self.tfadv: adv})
                print('loss: ', loss)
                actor_var_list=tf.contrib.framework.get_variables('res_pi/dense/bias:0')
                print(self.sess.run(actor_var_list))
                print(actor_var_list)

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]
    

    def _build_anet_ini(self, name, trainable, sigma_lower_bound):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            l1 = tf.layers.dense(l1, 100, tf.nn.relu, trainable=trainable)
            l1 = tf.layers.dense(l1, 100, tf.nn.relu, trainable=trainable)
            l1 = tf.layers.dense(l1, 100, tf.nn.tanh, trainable=trainable)

            # l1 = tf.layers.batch_normalization(tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable), training=True)
            '''the action mean mu is set to be scale 10 instead of 360, avoiding useless shaking and one-step to goal!'''
            mu =  30.*tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            # softplus as activation
            sigma = 0.01* tf.layers.dense(l1, A_DIM, tf.nn.sigmoid, trainable=trainable) # softplus to make it positive
            # in case that sigma is 0
            sigma +=sigma_lower_bound  #1e-4
            self.ini_mu=mu
            self.ini_sigma=sigma
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def _build_anet_res(self, name, trainable, sigma_lower_bound):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())
            l1 = tf.layers.dense(l1, 100, tf.nn.relu, trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())
            l1 = tf.layers.dense(l1, 100, tf.nn.relu, trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())
            l1 = tf.layers.dense(l1, 100, tf.nn.tanh, trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())

            # l1 = tf.layers.batch_normalization(tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable), training=True)
            '''the action mean mu is set to be scale 10 instead of 360, avoiding useless shaking and one-step to goal!'''
            mu =  30.*tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable, kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.zeros_initializer())
            # softplus as activation
            sigma = 0.01* tf.layers.dense(l1, A_DIM, tf.nn.sigmoid, trainable=trainable, kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.zeros_initializer()) # softplus to make it positive
            # in case that sigma is 0
            sigma +=sigma_lower_bound  #1e-4
            self.res_mu=mu
            self.res_sigma=sigma
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        res_mu, res_sigma, ini_mu, ini_sigma, a_ini, a_res, a  = self.sess.run([self.res_mu, self.res_sigma, self.ini_mu, self.ini_sigma, self.sample_ini_op, self.sample_res_op, self.sample_op ], {self.tfs: s})
        # print('s: ',s)
        # print('res: ', res)
        print('a_ini, a_res: ', a_ini, a_res)
        print('a: ', a)
        print('res_mu, res_sigma: ', res_mu,res_sigma)
        print('ini_mu, ini_sigma: ', ini_mu, ini_sigma)
        return np.clip(a[0], -360, 360)


    def pretrain_choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})
        # add noise, boosting exploration for updating critic
        a[0] += 10*np.random.rand(len(a[0]))
        return np.clip(a[0], -360, 360)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

env=Reacher(render=True)
ppo = PPO()
all_ep_r = []


'''pretrain the critic'''
# PRE_TRAIN = 8

# for ep in range(PRE_TRAIN):
#     print('pretrain step: {}'.format(ep))
#     s = env.reset()
#     # s=s/100.
#     buffer_s, buffer_a, buffer_r = [], [], []
#     ep_r = 0
#     for t in range(EP_LEN):    # in one episode
#         # env.render()
#         a = ppo.pretrain_choose_action(s)
#         # time.sleep(3)
#         s_, r, done = env.step(a)
#         # s_=s_/100.
#         buffer_s.append(s)
#         buffer_a.append(a)
#         # print('r, norm_r: ', r, (r+8)/8)
#         '''the normalization makes reacher's reward almost same and not work'''
#         # buffer_r.append((r+8)/8)    # normalize reward, find to be useful
#         buffer_r.append(r)
#         s = s_
#         ep_r += r

#         # update ppo
#         if (t+1) % BATCH == 0 or t == EP_LEN-1:
#             v_s_ = ppo.get_v(s_)
#             discounted_r = []
#             for r in buffer_r[::-1]:
#                 v_s_ = r + GAMMA * v_s_
#                 discounted_r.append(v_s_)
#             discounted_r.reverse()

#             bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
#             buffer_s, buffer_a, buffer_r = [], [], []
#             ppo.pretrain_critic(bs, ba, br)

# print('pretrain finished!')


for ep in range(EP_MAX):
    s = env.reset()
    # s=s/100.
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN):    # in one episode
        # env.render()
        print(s)
        a = ppo.choose_action(s)
        s_, r, done = env.step(a)
        # s_=s_/100.
        buffer_s.append(s)
        buffer_a.append(a)
        # print('r, norm_r: ', r, (r+8)/8)
        '''the normalization makes reacher's reward almost same and not work'''
        # buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        buffer_r.append(r)
        s = s_
        ep_r += r

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br)

    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )
    if ep % 50==0:
        plt.plot(np.arange(len(all_ep_r)), all_ep_r)
        plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.savefig('./ppo_single_ini.png')


print('all_ep_r: ',all_ep_r)