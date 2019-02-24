import argparse
import gym
import numpy as np
from itertools import count
from env import Reacher
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal,MultivariateNormal


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


# env = gym.make('CartPole-v0')
# env.seed(args.seed)
# torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden1_size)
        self.affine2 = nn.Linear(hidden1_size, hidden2_size)
        self.affine3_mu = nn.Linear(hidden2_size, output_size)
        self.affine3_sigma = nn.Linear(hidden2_size, output_size)
        self.hidden_acti=F.hardtanh
        self.output_acti=F.hardtanh
        # self.output_acti_sigma=nn.LeakyReLU(0.1)
        self.output_acti_sigma = F.hardtanh
        self.output_size=output_size

        self.saved_log_probs = []
        self.rewards = []
        self.entropy=[]

    def forward(self, x):
        x1 = self.hidden_acti(self.affine1(x))
        x2 = self.hidden_acti(self.affine2(x1))
        action_dis_mu = 360.*self.output_acti(self.affine3_mu(x2))
        # scale control action exploration noise, its a 1 dim tensor, should be learnable, but not here
        # scale = torch.from_numpy(np.array(self.output_size*[1.])).float()
        scale = self.output_acti_sigma(self.affine3_sigma(x2))

        return action_dis_mu, scale
        # return Normal(loc=action_dis_mu, scale=scale)

env=Reacher(render=True)
policy = Policy(env.num_observations, 100,200, env.num_actions)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()


def select_action(state):

    state = torch.from_numpy(state).float() # state: 2 dim tensor
    mu, scale =  policy(state)  
    # print('mu: ', mu)
    # print('scale: ',scale.squeeze())
    action_dis =Normal(loc=mu, scale=scale.squeeze())  # mu is 2d, scale is 1d
    # scale+=1e-6
    # action_dis =Normal(loc=mu, scale=scale) 

    action = action_dis.sample().squeeze()  # action: 1 dim tensor
    # print(action_dis.log_prob(action).squeeze()) # sometime nan, why?
    # print(action)
    policy.saved_log_probs.append(action_dis.log_prob(action).squeeze())  # also 1 dim tensor
    # print(action_dis.entropy())
    policy.entropy.append(action_dis.entropy())
    return action.numpy()


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    '''this normalization would make reward to have negative value'''
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob.double() * reward.double())
        # print('ou: ',-log_prob.double(), reward, -log_prob.double() * reward.double())
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()

    policy_loss.backward()
    optimizer.step()
    # for f in policy.parameters():
    #     print('data is')
    #     print(f.data)
    #     print('grad is')
    #     print(f.grad)
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    # running_reward = 10
    epi=[]
    epi_reward=[]
    training_episodes=10000
    for i_episode in range(training_episodes):
        state = env.reset()
        episode_reward=[]
        epi.append(i_episode)
        for t in range(10000):  # Don't infinite loop while learning
            # env.render() # display the env

            action = select_action(state)
            state, reward, done = env.step(action)

            if args.render:
                env.render()
            policy.rewards.append(reward)
            episode_reward.append(reward)
            if done:
                break
        epi_reward.append(np.sum(episode_reward))
        if i_episode%500 ==0:
            plt.plot(epi, epi_reward)
            plt.savefig('./reinforce.png')
        # running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        print('Episode: {},  Episode Reward: {}'.format(i_episode,np.sum(episode_reward)))
        # if i_episode % args.log_interval == 0:
        #     print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
        #         i_episode, t, running_reward))
        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break


if __name__ == '__main__':
    main()