# Supervised Learning with Samples from Inverse Kinematics

The training is actually not stable for different running!

* `inverse_kinematics_demo_new.py`: use inverse kinematics to reach the goal position and derive the corresponding joint angles, then generate straight line trajectories in theta (joint angles) space. Generated data file is `data10.p` .
* `inverse_kinematics_demo_ini2goal.py`: generate trajectories like above, but from arbitrary initial positions to arbitrary goal positions. (for training to grasp the moving mechanism of reacher)
* `inverse_kinematics_demo.py`: generate trajectories directly using inverse kinematics, it's not an accurate straight line in x-y space as inverse kinematics contains approximations.

