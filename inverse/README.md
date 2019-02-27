# Inverse Kinematics Data Generation

- `inverse_kinematics_demo_new.py`: generate straight line trajectories in theta space. In order to do that, it needs to first reach the target position for each episode with inverse kinematics, and give the joint angles  (may be multivalued) at the target position. Then it calculates the difference of initial joint angles and joint angles at target position, and divide it to be several steps of actions, which form a straight line trajectories in the theta (joint angle) space.
- `inverse_kinematics_demo_test.py`: generate same trajectories as inverse kinematics shown (inverse kinematics includes approximation to line trajectories in x-y space) .
