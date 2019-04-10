# DDPG with Residual Policy Learning

## Description:

Train a residual policy with RL (DDPG) on top of a pre-trained initialization policy via supervised learning with samples generated from inverse kinematics (already generated).

## To Use:

1. Pre-train the initialization policy in `./ini` with `python predict_test.py --train`;
2. Train the initialized policy with RL: `python -m run --alg=ddpg --num_timesteps=3000 --train`.

