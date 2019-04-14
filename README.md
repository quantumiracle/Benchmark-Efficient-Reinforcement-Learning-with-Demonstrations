# Benchmark Efficient Reinforcement Learning with/without Demonstrations
Efficient reinforcement learning for robotics control in simulation (_Reacher_ Environment).

Compare present methods for more efficient and robust RL training, including:

* Better Initialization (pre-trained with supervised learning);
* Residual Policy Learning;
* DDPG from Demonstrations;
* Reptile, MAML (across tasks);

## To Run:

* Python 3.5

* Pytorch & Tensorflow

## Benchmarks:

* DDPG, PPO for Reacher Environment in simulation
* Inverse Kinematics of Reacher Environment
* Supervised learning for intialization of DDPG, PPO
* Residual policy learning for intialization of DDPG, PPO
* Reptile + PPO, MAML + PPO
* DDPG from demonstrations

## Contents:

### Basics:

1. `./origin_env_code`:

* Basic codes for Reacher environment and inverse kinematics.

2. `./DDPG4Reacher`, `./DDPG4Reacher2`:

* DDPG algorithm for Reacher environment.

3. `./Inverse`:

* Inverse kinematics for generating demonstrations data.

### Efficient RL:

1. **Policy Replacement (Behavior Cloning)** in `./DDPG_Inverse`:

* Train an initialization policy for RL (DDPG) via supervised learning with samples generated from inverse kinematics (already generated).

2. **Feeding Demonstrations into Memory Buffer** in `./DDPGfD`:

- DDPGfD codes are training DDPG to learn from demonstrations, feeding demonstration trajectories directly into memory (a separate one) for training.

3. **Residual Policy Learning** in `./RPL_DDPG_new/`:

* Train a residual policy with RL (DDPG) on top of a pre-trained initialization policy via supervised learning with samples generated from inverse kinematics (already generated).

4. **Meta-Learning and Policy Replacement with PPO** in `./PPO`:

* Implementations of PPO algorithm with Reacher environment, including PPO for Reacher of 2/3 joints;
* PPO with initialized policy;
* PPO+Reptile;
*  PPO+MAML;
*  PPO+FOMAML (first-order MAML), etc.

### Comparison:

`./Comparison`:

* Comparison of different methods with demonstrations for efficient reinforcement learning (DDPG), including policy replacement (`./Comparison/DDPGini/`), residual policy learning (`./Comparison/DDPGres/`), directly feeding demonstrations (demonstration ratio: 0.5, (`./Comparison/DDPGfD/`)) into the buffer and vanilla DDPG.

**Dense Reward:**

<p align="center">
<img src="https://github.com/quantumiracle/Reinforcement-Learning-for-Robotics/blob/master/img/3000step41.png" width="80%">

**Sparse Reward:**

<p align="center">
<img src="https://github.com/quantumiracle/Reinforcement-Learning-for-Robotics/blob/master/img/3000step_sparse2.png" width="80%">
