import gym
import torch
import multiprocessing as mp

from maml_rl.envs.subproc_vec_env1 import SubprocVecEnv
from maml_rl.episode import BatchEpisodes
from maml_rl.envs.reacher.env import Reacher_target
import numpy as np

# def make_env(env_name):
#     def _make_env():
#         return gym.make(env_name)
#     return _make_env
def make_env(env_name):
    screen_size = 1000
    link_lengths = [200, 140, 100]
    joint_angles = [0, 0, 0]
    target_pos=[369,430]
    if env_name=='reacher':
        # reacher=Reacher_target(screen_size, link_lengths, joint_angles, target_pos)
        reacher=Reacher_target({'target': [700.0,100.0]})
    return reacher

class BatchSampler(object):
    def __init__(self, env_name, batch_size, num_workers=mp.cpu_count() - 1):
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.queue = mp.Queue()
        '''need to understand more'''
        self.envs = SubprocVecEnv([make_env(env_name) for _ in range(num_workers)],
            queue=self.queue)
        # self._env = gym.make(env_name)
        self._env=make_env(env_name)

    def sample(self, policy, params=None, gamma=0.95, device='cpu'):
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        
        observations, batch_ids = self.envs.reset()
        '''to float'''
        observations=observations.astype(np.float32)

        dones = [False]
        # when batch_size is full, not self.queue.empty() automatically change false
        # one batch contains a full episode (until done) of data
        while (not all(dones)) or (not self.queue.empty()):
        # while (not all(dones)) :
            # print(not all(dones),(not self.queue.empty()))
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=device)
                print('obs: ',observations)
                '''.float'''
                # .sample() is sample from distribution, output of NN is formed into a normal distribution
                actions_tensor = policy(observations_tensor.float(), params=params).sample()
                actions = actions_tensor.cpu().numpy()

            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            print('batch_ids: ', new_batch_ids)
            print('rewards: ', rewards)
            episodes.append(observations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids
            # print('actions: {}'.format(actions))
            # print('observations: {}'.format(new_observations))
            # print('rewards: {}'.format(rewards))

            
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return episodes

    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks):
        tasks = self._env.sample_tasks(num_tasks)
        return tasks
