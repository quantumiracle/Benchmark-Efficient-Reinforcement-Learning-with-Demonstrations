import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
from env import Reacher
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical,Normal


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
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


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden1_size)
        self.action_head_mu = nn.Linear(hidden1_size, output_size)
        self.action_head_sigma = nn.Linear(hidden1_size, output_size)
        self.value_head = nn.Linear(hidden1_size, 1)
        self.output_acti_mu=F.hardtanh
        self.output_acti_sigma=nn.LeakyReLU(0.1)
        self.output_size=output_size

        self.saved_actions = []
        self.rewards = []

    def softplus(self, x):
        return torch.log(1.+torch.exp(x))

    def forward(self, x):
        x = F.relu(self.affine1(x))
        state_values = self.value_head(x)
        action_dis_mu = 360.*self.output_acti_mu(self.action_head_mu(x))
        action_dis_sigma = self.output_acti_sigma(self.action_head_sigma(x))
        # print(action_dis_sigma)
        # scale = torch.from_numpy(np.array(self.output_size*[5.])).float()
        
        # return Normal(loc=action_dis_mu, scale=action_dis_sigma), state_values
        return action_dis_mu,action_dis_sigma,state_values

env=Reacher(render=True)
model = Policy(env.num_observations, 100,200, env.num_actions)
optimizer = optim.Adam(model.parameters(), lr=3e-1)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    mu, scale, state_value = model(state)
    action_dis = Normal(loc=mu, scale=scale.squeeze())
    action = action_dis.sample().squeeze()
    model.saved_actions.append(SavedAction(action_dis.log_prob(action).squeeze(), state_value))
    return action.numpy()


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    training_episodes=100000
    epi=[]
    epi_reward=[]
    for i_episode in range(training_episodes):
        epi.append(i_episode)
        state = env.reset()
        episode_reward=[]
        for t in range(10000):  # Don't infinite loop while learning

            action = select_action(state)
            state, reward, done = env.step(action)

            if args.render:
                env.render()
            model.rewards.append(reward)
            episode_reward.append(reward)


            if done:
                break
        epi_reward.append(np.sum(episode_reward))
        if i_episode%500 ==0:
            plt.plot(epi, epi_reward)
            plt.savefig('./actor_critic.png')
        finish_episode()
        print('Episode: {},  Episode Reward: {}'.format(i_episode,np.sum(episode_reward)))


if __name__ == '__main__':
    main()