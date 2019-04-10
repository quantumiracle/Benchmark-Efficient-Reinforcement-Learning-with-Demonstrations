# Supervised Learning with Samples from Inverse Kinematics

## Description:

Train an initialization policy for RL (DDPG) via supervised learning with samples generated from inverse kinematics (already generated).

## Contents:

* predict.py: first version of code
* predict_ppo.py: try to used as initialize the openai version ppo (fail)
* predict_test.py: final code, train with epoch data.
* predict_test_batch.py: former wrong code, train with batch data.
* predict_test_norm_output.py: final code, train with epoch and normalize the output of NN (action) to be -1~1, need RL to change accordingly (action scale).
* env_test.py: reacher env.

## To Use:

`python predict_test.py --train`

