# DDPG from Demonstration

- `env_2obstacle.py`:  reacher env with 2 obstacles in half way of the optimal trajectory from initial position (joint angles: [0.1, 0.1, 0.1]) to final goal, the state space is 8 dim, without target position.
- `env_2obstacle_10dim.py`: reacher env with 2 obstacles in half way of the optimal trajectory from initial position (joint angles: [0.1, 0.1, 0.1]) to final goal, the state space is 10 dim, with target position.
- `inverse_kinematics_demon_2obstacle_noise_memory.py`: generate full demonstrations ([state, action ,reward, new_state, done]) to feed in ddpg memory in file `data_memory2.p` with env `env_2obstacle_10dim.py`.
- `inverse_kinematics_demon_2obstacle_noise.py`: generate noisy demonstrations ([state, action]) in data file to train the nn in `predict_test.py`.
- `inverse_kinematics_demon_2obstacle.py`: generate no-noise demonstrations  ([state, action]) in data file to train the nn in `predict_test.py`.
- `data_memory2`: data file generated from `inverse_kinematics_demon_2obstacle_noise_memory.py` with 50 episodes and 20 steps per episode.
- DDPGfD codes are training DDPG to learn from demonstrations, feeding demonstration trajectories into memory.
