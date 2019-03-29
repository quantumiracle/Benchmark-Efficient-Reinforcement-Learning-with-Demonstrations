import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
from env_2 import Reacher_for2 as Reacher
from copy import copy
import argparse
ITR=1000  # number of tasks
EP_MAX = 20  # number of episodes for single task, generally 2000 steps for 2 joints and 5000 steps for 3 joints have a good performance
EP_LEN = 20  # number of steps for each episode
GAMMA = 0.9
A_LR = 1e-4
C_LR = 2e-4
BATCH = 64
A_UPDATE_STEPS = 3
C_UPDATE_STEPS = 3
S_DIM, A_DIM = 8,2
TRAIN_INTERVAL = 10

METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization

model_path='./maml_model/maml'

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)

args = parser.parse_args()


class PPO(object):

    def __init__(self):
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        self.sess = tf.Session(config=config)
        # self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        self.critic_weights = {}
        self.actor_weights = {}
        self.actor_weights_ = {}
        self.construct_weights()
        self.forward(self.actor_weights, self.actor_weights_, self.critic_weights)
        self.sess.run(tf.global_variables_initializer())
        self.actor_var = [v for v in tf.trainable_variables() if v.name.split('/')[0] == "pi"]
        self.critic_var= [v for v in tf.trainable_variables() if v.name.split('/')[0] == "critic"]
        
        tf.summary.FileWriter("log/", self.sess.graph)
        
    def forward(self, actor_weights, actor_weights_, critic_weights):
        '''
        forward process, including loss function defination and optimization operations
        '''
        # critic
        with tf.variable_scope('critic'):
            # l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            # self.v = tf.layers.dense(l1, 1)
            self.v = self._build_cnet('critic', weights = critic_weights)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            with tf.variable_scope('adam', reuse = tf.AUTO_REUSE):
                self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)
        self.critic_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')
        # print('cirtic params: ', self.critic_params)

        # actor
        self.pi, self.pi_params = self._build_anet('pi', trainable=True, weights = actor_weights)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False, weights = actor_weights_)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(self.pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(self.pi_params, oldpi_params)]


        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = self.pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, self.pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))
            # add entropy to boost exploration
            # entropy=pi.entropy()
            # self.aloss-=0.1*entropy
        with tf.variable_scope('atrain'):
            with tf.variable_scope('adam', reuse = tf.AUTO_REUSE):
                self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)


        

    def construct_weights(self, ):
        '''
        define weights
        '''
        c_hidden1 = 100
        a_hidden1 = 100

        with tf.variable_scope('critic'):
            self.critic_weights['w1'] = tf.Variable(tf.truncated_normal([S_DIM, c_hidden1], stddev = 0.1))
            self.critic_weights['b1'] = tf.Variable(tf.truncated_normal([c_hidden1], stddev = 0.1))
            self.critic_weights['w2'] = tf.Variable(tf.truncated_normal([c_hidden1, 1], stddev = 0.1))
            self.critic_weights['b2'] = tf.Variable(tf.truncated_normal([1], stddev = 0.1))

        with tf.variable_scope('pi'):
            self.actor_weights['w1'] = tf.Variable(tf.truncated_normal([S_DIM, a_hidden1], stddev = 0.1), trainable = True)
            self.actor_weights['b1'] = tf.Variable(tf.truncated_normal([a_hidden1], stddev = 0.1), trainable = True)
            self.actor_weights['w2'] = tf.Variable(tf.truncated_normal([a_hidden1, A_DIM], stddev = 0.1), trainable = True)
            self.actor_weights['b2'] = tf.Variable(tf.truncated_normal([A_DIM], stddev = 0.1), trainable = True)
            self.actor_weights['w3'] = tf.Variable(tf.truncated_normal([a_hidden1, A_DIM], stddev = 0.1), trainable = True)
            self.actor_weights['b3'] = tf.Variable(tf.truncated_normal([A_DIM], stddev = 0.1), trainable = True)

        with tf.variable_scope('oldpi'):
            self.actor_weights_['w1'] = tf.Variable(tf.truncated_normal([S_DIM, a_hidden1], stddev = 0.1), trainable = False)
            self.actor_weights_['b1'] = tf.Variable(tf.truncated_normal([a_hidden1], stddev = 0.1), trainable = False)
            self.actor_weights_['w2'] = tf.Variable(tf.truncated_normal([a_hidden1, A_DIM], stddev = 0.1), trainable = False)
            self.actor_weights_['b2'] = tf.Variable(tf.truncated_normal([A_DIM], stddev = 0.1), trainable = False)
            self.actor_weights_['w3'] = tf.Variable(tf.truncated_normal([a_hidden1, A_DIM], stddev = 0.1), trainable = False)
            self.actor_weights_['b3'] = tf.Variable(tf.truncated_normal([A_DIM], stddev = 0.1), trainable = False)
            



    def update(self, s, a, r):
        '''
        meta policy update
        '''
        self.forward(self.new_a_weights, self.new_a_weights, self.new_c_weights)
        self.sess.run(self.update_oldpi_op)
        print(r)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        print(adv)
        adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for i in range(A_UPDATE_STEPS):
                # print('updata: ',i)
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
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]
        # print([v.name for v in self.critic_var],self.sess.run(self.critic_var))
        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def sum_gradients_update(self, s, a, r, stepsize):
        '''
        task specific policy update, not directly update the weights but derive the new weights
        '''
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for i in range(A_UPDATE_STEPS):
                # print('updata: ',i)
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
            
            # self.tfs = s
            # self.tfa = a
            # self.tfadv = adv
            a_grads_form  = tf.gradients(self.aloss, list(self.actor_weights.values()))
            # a_grads_value = a_grads_form.eval(feed_dict={self.tfs: s, self.tfa: a, self.tfadv: adv})
            a_grads = dict(zip(self.actor_weights.keys(), a_grads_form))
            self.new_a_weights = dict(zip(self.actor_weights.keys(), [self.actor_weights[key] - stepsize * a_grads[key] 
            for key in self.actor_weights.keys()]))
            # self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
            # self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
            # self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

            # print(self.critic_var)
            # print(self.actor_var)
        # print([v.name for v in self.critic_var],self.sess.run(self.critic_var))


        # update critic
        # [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]
        
        # self.tfs = s
        # self.tfdc_r = r
        c_grads_form = tf.gradients(self.closs, list(self.critic_weights.values()))
        # c_grads_value = c_grads_form.eval()
        c_grads = dict(zip(self.critic_weights.keys(), c_grads_form))
        self.new_c_weights = dict(zip(self.critic_weights.keys(), [self.critic_weights[key] - stepsize * c_grads[key] 
        for key in self.critic_weights.keys()]))
        # self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        # self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # self.actor_weights = new_a_weights
        # self.critic_weights = new_c_weights
        # return new_a_weights, new_c_weights

    def _build_anet(self, name, trainable, weights):
        '''
        build the actor network with defined weights
        '''
        with tf.variable_scope(name):

            # l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            # # l1 = tf.layers.batch_normalization(tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable), training=True)
            # '''the action mean mu is set to be scale 10 instead of 360, avoiding useless shaking and one-step to goal!'''
            # mu =  10.*tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            # sigma = tf.layers.dense(l1, A_DIM, tf.nn.sigmoid, trainable=trainable) # softplus to make it positive
            
            a_l1 = tf.nn.relu(tf.matmul(self.tfs, weights['w1'])+weights['b1'])
            mu = 10.*tf.nn.tanh(tf.matmul(a_l1, weights['w2'])+weights['b2'])
            sigma = tf.nn.sigmoid(tf.matmul(a_l1, weights['w3'])+weights['b3'])

            # in case that sigma is 0
            sigma +=1e-4
            self.mu=mu
            self.sigma=sigma
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def _build_cnet(self, name, weights):
        '''
        build the critic network with defined weights
        '''
        with tf.variable_scope(name):
            c_l1 = tf.nn.relu(tf.matmul(self.tfs, weights['w1']) + weights['b1'])
            output = tf.matmul(c_l1, weights['w2']) + weights['b2']
        return output


    def choose_action(self, s):
        
        s = s[np.newaxis, :]
        # a ,mu, sigma= self.sess.run([self.sample_op, self.mu, self.sigma], {self.tfs: s})
        a= self.sess.run(self.sample_op, {self.tfs: s})
        # print('s: ',s)
        # print('a: ', a)
        # print('mu, sigma: ', mu,sigma)
        return np.clip(a[0], -360, 360)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]
    def save_model(self, ):
        self.actor_weights=self.sess.run(self.actor_var)
        self.critic_weights=self.sess.run(self.critic_var)
        return self.actor_weights, self.critic_weights
    # def value(self, ):
    #     return self.sess.run(self.actor_weights_before)[-1]

    def restore_model(self, a_te, c_te):
        # restore the actor
        with tf.variable_scope('pi'):
            self.restore_pi = [pi.assign(te) for pi, te in zip(self.pi_params, np.array(a_te))]
        self.sess.run(self.restore_pi)

        self.sess.run(self.update_oldpi_op)

        # restore the critic

        with tf.variable_scope('critic'):
            self.restore_critic =[cri.assign(te) for cri, te in zip(self.critic_params, np.array(c_te))]
        self.sess.run(self.restore_critic)
    
    def save(self, path):
            saver = tf.train.Saver()
            saver.save(self.sess, path)

    def load(self, path):
            saver=tf.train.Saver()
            saver.restore(self.sess, path)




    


def sample_task():
    range_pose=0.3
    target_pose=(2*np.random.rand(2)-1)*range_pose + [0.5, 0.5] # around center (0.5,0.5), range 0.3
    screen_size=1000
    target_pose=target_pose*screen_size

    
    env=Reacher(target_pos=target_pose, render=True)
    return env, target_pose

def meta_update(ppo):
    train_set_a = []
    train_set_s = []
    train_set_r = []
    for ep in range(TRAIN_INTERVAL):
        s = env.reset()
        s=s/100. # scale the inputs
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        for t in range(EP_LEN):    #  steps in one episode
            # env.render()
            a = ppo.choose_action(s)
            s_, r, done = env.step(a)
            s_=s_/100.
            buffer_s.append(s)
            buffer_a.append(a)
            # print('r, norm_r: ', r, (r+8)/8)
            '''the normalization makes reacher's reward almost same and not work'''
            # buffer_r.append((r+8)/8)    # normalize reward, find to be useful
            buffer_r.append(r)
            s = s_
            ep_r += r

            # update ppo
            if ((t+1) % BATCH == 0 or t == EP_LEN-1):
                v_s_ = ppo.get_v(s_)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()

                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                train_set_a.append(ba)
                train_set_r.append(br)
                train_set_s.append(bs)
                # print(bs)
    train_set_a = np.array(train_set_a).reshape(-1, len(ba[-1]))
    train_set_s = np.array(train_set_s).reshape(-1, len(bs[-1]))
    train_set_r = np.array(train_set_r).reshape(-1, len(br[-1]))
    ppo.update(train_set_s, train_set_a, train_set_r)
    return ppo


ppo = PPO()

if args.train:

    # env=Reacher(render=True)
    stepsize0=0.01
    test_env, t=sample_task()
    itr_test_rewards=[]
    for itr in range (ITR):
        # randomly sample a task (different target position)
        np.random.seed(itr)
        env, target_position =sample_task()
        print('Task {}: target position {}'.format(itr+1, target_position))
        all_ep_r = []
        train_set_a = []
        train_set_s = []
        train_set_r = []
        # inner policy update
        for ep in range(EP_MAX):  # EP_MAX: how many episodes for training one tasks
            # print(actor_weights_before[-1])
            s = env.reset()
            s=s/100. # scale the inputs
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            for t in range(EP_LEN):    #  steps in one episode
                # env.render()
                a = ppo.choose_action(s)
                s_, r, done = env.step(a)
                s_=s_/100.
                buffer_s.append(s)
                buffer_a.append(a)
                # print('r, norm_r: ', r, (r+8)/8)
                '''the normalization makes reacher's reward almost same and not work'''
                # buffer_r.append((r+8)/8)    # normalize reward, find to be useful
                buffer_r.append(r)
                s = s_
                ep_r += r

                # update ppo
                if ((t+1) % BATCH == 0 or t == EP_LEN-1):
                    v_s_ = ppo.get_v(s_)
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    train_set_a.append(ba)
                    train_set_r.append(br)
                    train_set_s.append(bs)

                    # ppo.update(bs, ba, br)

            if ep%TRAIN_INTERVAL == 0:
                train_set_a = np.array(train_set_a).reshape(-1, len(ba[-1]))
                train_set_s = np.array(train_set_s).reshape(-1, len(bs[-1]))
                train_set_r = np.array(train_set_r).reshape(-1, len(br[-1]))
                print('inner policy update begin')
                ppo.sum_gradients_update(train_set_s, train_set_a, train_set_r, stepsize0)
                print('inner policy update finish')
                print('meta policy update begin')
                ppo = meta_update(ppo)
                print('meta policy update finish')
                train_set_a = []
                train_set_s = []
                train_set_r = []


            if ep == 0: all_ep_r.append(ep_r)
            else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
            print(
                'Ep: %i' % ep,
                "|Ep_r: %.2f" % ep_r,
                ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
            )
            # if ep % 500==0:
            #     plt.plot(np.arange(len(all_ep_r)), all_ep_r)
            #     plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.savefig('./ppo_reptile.png')
            
        # meta policy update, same as inner loop for FOMAML



        # stepsize=stepsize0*(1-itr/ITR)  # decayed learning rate/step size, so not learn after several steps.


        # test 1 episode on test_env
        actor_weights_test, critic_weights_test= ppo.save_model()  # save the model and restore it after test
        all_ep_r = []
        print('-------------- TEST --------------- ')
        for ep in range(EP_MAX):
            s = test_env.reset()
            s=s/100. # scale the inputs
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            for t in range(EP_LEN):    # in one episode
                a = ppo.choose_action(s)
                s_, r, done = test_env.step(a)
                s_=s_/100.
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
            all_ep_r.append(ep_r)
            # if ep == 0: all_ep_r.append(ep_r)
            # else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
            # print(
            #     'Ep: %i' % ep,
            #     "|Ep_r: %i" % ep_r,
            #     ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
            # )
            # if ep % 500==0:
            #     plt.plot(np.arange(len(all_ep_r)), all_ep_r)
            #     plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.savefig('./ppo_reptile.png')
        # restore before test
        ppo.restore_model(actor_weights_test, critic_weights_test)
        itr_test_rewards.append(np.average(np.array(all_ep_r)))
        plt.plot(np.arange(len(itr_test_rewards)), itr_test_rewards)
        plt.savefig('./ppo_maml.png')
        if itr%10 == 0:
            ppo.save(model_path)

if args.test:

    stepsize0=0.1
    test_env, t=sample_task()
    all_ep_r = []
    ppo.load(model_path)
    print('-------------- TEST --------------- ')
    for ep in range(EP_MAX):
        print('Episode: ', ep)
        s = test_env.reset()
        s=s/100. # scale the inputs
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        for t in range(EP_LEN):    # in one episode
            a = ppo.choose_action(s)
            s_, r, done = test_env.step(a)
            s_=s_/100.
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
        all_ep_r.append(ep_r)
        # if ep == 0: all_ep_r.append(ep_r)
        # else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
        # print(
        #     'Ep: %i' % ep,
        #     "|Ep_r: %i" % ep_r,
        #     ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
        # )
        # if ep % 500==0:
        #     plt.plot(np.arange(len(all_ep_r)), all_ep_r)
        #     plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.savefig('./ppo_reptile.png')
    # restore before test

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.savefig('./ppo_maml_test.png')