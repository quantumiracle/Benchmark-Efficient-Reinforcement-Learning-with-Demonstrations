# Inverse Kinematics Data Generation (for Supervised Learning)

The training is actually not stable for different running!

- `inverse_kinematics_demo_new.py`: generate straight line trajectories in theta space. In order to do that, it needs to first reach the target position for each episode with inverse kinematics, and give the joint angles  (may be multivalued) at the target position. Then it calculates the difference of initial joint angles and joint angles at target position, and divide it to be several steps of actions, which form a straight line trajectories in the theta (joint angle) space. Generated data file is `data10.p` .
- `inverse_kinematics_demo_test.py`: generate same trajectories as inverse kinematics shown (inverse kinematics includes approximation to line trajectories in x-y space) .
- `inverse_kinematics_demo_ini2goal.py`: generate trajectories like above, but from arbitrary initial positions to arbitrary goal positions. (for training to grasp the moving mechanism of reacher)
- `inverse_kinematics_demo.py`: generate trajectories directly using inverse kinematics, it's not an accurate straight line in x-y space as inverse kinematics contains approximations.

