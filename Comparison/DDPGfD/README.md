# Comparisons--Feeding Demonstrations into Memory Buffer:

Comparison of different methods with demonstrations for efficient reinforcement learning (DDPG), including policy replacement, residual policy learning, directly feeding demonstrations (demonstration ratio: 0.5) into the buffer and vanilla DDPG.

* Environment: the leanring environment is a Reacher environment with 2 penalty areas of radius 50 and joint length of [200, 140, 100] with the screen size 1000, value range of joint angles is [-360,360].
* Neural networks: the actor network includes 5 fully connected layers with ReLu as hidden layer activation function and Tanh as output activation, and the size of each layer is 100 for the first 4 layer and 400 for the last layer. The critic network are almost the same just without activation for the last layer and so with Tanh for the last but one layers. The learning rate is $10^{-4}$ for the actor and $10^{-3}$ for the critic. Batchsize is 640, and exploration uses normal noise of scale 2.0 for experiments. Input dimension of policy network is 10 (4 pairs of joint positions including the (0,0), 1 pair of target position), and output dimension is 3 action values for each joints.
* Dataset of demonstrations: we use two datasets of demonstrations, a small one and a large one. The small dataset contains 50 episodes of expert trajectories and the large one contains 1000 episodes, with 21 steps for each episodes. The demonstration trajectories are generated with a intermediate goal at between the two penalty areas via the inverse kinematics, and with injected normal noise. The demonstration data is used in several methods, including learning from demonstrations, pre-training the policy network as initialization and pre-training the policy network in residual learning.
* Training: both the initialization and residual learning methods are set to have a pre-heating process for training the critic, and the length of the pre-heating is 600 steps. The number of overall training steps is set to be $10^4$ for all methods.

## To Use:

`python -m run --alg=ddpg --num_timesteps=3000 --train ` to train the DDPG with demonstrations fed directly into a buffer for training.

