# from __future__ import division
import pygame
import numpy as np
import numdifftools as nd
import time
import pickle

'''
generate one sample demonstration with no noise for fixed target position
with a two step reaching for env with obstacle, one intermediate goal and one targer goal
'''

# Function to compute the transformation matrix between two frames, based on the length (in y-direction) along the first frame, and the rotation of the second frame with respect to the first frame
def compute_transformation(length, rotation):
    cos_rotation = np.cos(rotation)
    sin_rotation = np.sin(rotation)
    transformation = np.array([[cos_rotation, -sin_rotation, 0], [sin_rotation, cos_rotation, length], [0, 0, 1]])
    return transformation


# Function to compute the pose of end of the robot, based on the angles of all the joints
def compute_end_pose(current_joint_angles):
    trans_10 = compute_transformation(0, current_joint_angles[0])  # This matrix transforms the origin of joint 1 into the world frame
    trans_21 = compute_transformation(link_lengths[0], current_joint_angles[1])  # This matrix transforms the origin of joint 2 into the frame of joint 1
    trans_32 = compute_transformation(link_lengths[1], current_joint_angles[2])  # This matrix transforms the origin of joint 3 into the frame of joint 2
    trans_30 = np.dot(trans_10, np.dot(trans_21, trans_32))  # This matrix transforms a point in the frame of joint 3 into the world frame
    end_pose = np.dot(trans_30, np.array([0, link_lengths[2], 1]))  # This is the position of the robot's end in the world frame
    return end_pose[0:2]  # Only take the x and y components, not the w component


# Function to compute the Jacobian, representing the rate of change of the robot's end pose with the robot's joint angles, computed at the current robot pose
def compute_jacobian(current_joint_angles):
    jacobian = nd.Jacobian(compute_end_pose)([current_joint_angles[0], current_joint_angles[1], current_joint_angles[2]])
    return jacobian

'joint_angles time link_length and time screen_size to contributes to visulizaiton, while in env_test.py only times link_length'
# Function to draw the state of the world onto the screen
def draw_current_state(current_joint_angles, current_target_pose): 
    # First, compute the transformation matrices for the frames for the different joints
    trans_10 = compute_transformation(0, current_joint_angles[0])
    trans_21 = compute_transformation(link_lengths[0], current_joint_angles[1])
    trans_32 = compute_transformation(link_lengths[1], current_joint_angles[2])
    # Then, compute the coordinates of the joints in world space
    joint_1 = np.dot(trans_10, np.array([0, 0, 1]))
    joint_2 = np.dot(trans_10, (np.dot(trans_21, np.array([0, 0, 1]))))
    joint_3 = np.dot(trans_10, (np.dot(trans_21, np.dot(trans_32, np.array([0, 0, 1])))))
    robot_end = np.dot(trans_10, (np.dot(trans_21, np.dot(trans_32, np.array([0, link_lengths[2], 1])))))
    # Then, compute the coordinates of the joints in screen space
    joint_1_screen = [int((0.5 + joint_1[0]) * screen_size), int((0.5 - joint_1[1]) * screen_size)]
    joint_2_screen = [int((0.5 + joint_2[0]) * screen_size), int((0.5 - joint_2[1]) * screen_size)]
    joint_3_screen = [int((0.5 + joint_3[0]) * screen_size), int((0.5 - joint_3[1]) * screen_size)]
    robot_end_screen = [int((0.5 + robot_end[0]) * screen_size), int((0.5 - robot_end[1]) * screen_size)]
    target_screen = [int((0.5 + current_target_pose[0]) * screen_size), int((0.5 - current_target_pose[1]) * screen_size)]
    # Finally, draw the joints and the links
    screen.fill((0, 0, 0))
    pygame.draw.circle(screen, (255, 0, 0), target_screen, 15)
    pygame.draw.line(screen, (255, 255, 255), joint_1_screen, joint_2_screen, 5)
    pygame.draw.line(screen, (255, 255, 255), joint_2_screen, joint_3_screen, 5)
    pygame.draw.line(screen, (255, 255, 255), joint_3_screen, robot_end_screen, 5)
    pygame.draw.circle(screen, (0, 0, 255), joint_1_screen, 10)
    pygame.draw.circle(screen, (0, 0, 255), joint_2_screen, 10)
    pygame.draw.circle(screen, (0, 0, 255), joint_3_screen, 10)
    pygame.draw.circle(screen, (0, 255, 0), robot_end_screen, 10)

    pygame.draw.circle(screen, (255, 255, 0), np.array(current_target_pose).astype(int), 10)
    ''' show the graph'''
    pygame.display.flip()
    time.sleep(0.01)



    return np.array([joint_1_screen,joint_2_screen,joint_3_screen,robot_end_screen, target_screen]).reshape(-1)

def get_state(current_joint_angles, current_target_pose, reshape = True):
        # First, compute the transformation matrices for the frames for the different joints
    trans_10 = compute_transformation(0, current_joint_angles[0])
    trans_21 = compute_transformation(link_lengths[0], current_joint_angles[1])
    trans_32 = compute_transformation(link_lengths[1], current_joint_angles[2])
    # Then, compute the coordinates of the joints in world space
    joint_1 = np.dot(trans_10, np.array([0, 0, 1]))
    joint_2 = np.dot(trans_10, (np.dot(trans_21, np.array([0, 0, 1]))))
    joint_3 = np.dot(trans_10, (np.dot(trans_21, np.dot(trans_32, np.array([0, 0, 1])))))
    robot_end = np.dot(trans_10, (np.dot(trans_21, np.dot(trans_32, np.array([0, link_lengths[2], 1])))))
    # Then, compute the coordinates of the joints in screen space
    joint_1_screen = [int((0.5 + joint_1[0]) * screen_size), int((0.5 - joint_1[1]) * screen_size)]
    joint_2_screen = [int((0.5 + joint_2[0]) * screen_size), int((0.5 - joint_2[1]) * screen_size)]
    joint_3_screen = [int((0.5 + joint_3[0]) * screen_size), int((0.5 - joint_3[1]) * screen_size)]
    robot_end_screen = [int((0.5 + robot_end[0]) * screen_size), int((0.5 - robot_end[1]) * screen_size)]
    target_screen = [int((0.5 + current_target_pose[0]) * screen_size), int((0.5 - current_target_pose[1]) * screen_size)]
    if reshape:
        return np.array([joint_1_screen,joint_2_screen,joint_3_screen,robot_end_screen, target_screen]).reshape(-1)
    else:
        return np.array([joint_1_screen,joint_2_screen,joint_3_screen,robot_end_screen, target_screen])

# normalize the joint angle to be in range [-2pi,2pi]
def norm_angle(angle):
    while angle>2*np.pi:
        angle-=2*np.pi
    while angle<-2*np.pi:
        angle+=2*np.pi
    return angle

# 2 obstacles
screen_size=1000
link_lengths = [0.2, 0.15, 0.1]  # These lengths are in world space (0 to 1), not screen space (0 to 1000)
step_size = 0.1  # This is the gradient descent step taken when solving the inverse kinematics

# target goal for screen 0-1000
TARGET_GOAL0 = np.array([screen_size/4, screen_size*3/4])
# converted target goal for screen -0.5-0.5 (for inverse kinematics)
TARGET_GOAL = np.array([TARGET_GOAL0[0]-500, -TARGET_GOAL0[1]+500]) / screen_size

start_joint_angles = [0.1,0.1,0.1]
start_end_pos = get_state(start_joint_angles, TARGET_GOAL, reshape = False)[3]  # initial end of joint
print('start: ', start_end_pos)


OBSTACLE_RADIUS = 50
OBSTACLE_DISTANCE = 180
NUM_OBSTACLES = 2
obstacle1_pos=0.5 * ( np.array(start_end_pos)+np.array(TARGET_GOAL0))
obstacle2_pos=0.5 * ( np.array(start_end_pos)+np.array(TARGET_GOAL0)) - np.array([OBSTACLE_DISTANCE,0])

# use intermediate position of two obstacles as intermediate goal position
# intermediate goal for screen 0-1000
INTER_GOAL0 = 0.5 * ( np.array(start_end_pos)+np.array(TARGET_GOAL0)) - 0.5 * np.array([OBSTACLE_DISTANCE,0])
# converted intermediate goal for screen -0.5-0.5 (for inverse kinematics)
INTER_GOAL = np.array([INTER_GOAL0[0]-500, -INTER_GOAL0[1]+500]) / screen_size




# Create a PyGame instance
screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption("Inverse Kinematics Demo")
pygame.display.flip()
time.sleep(0.2)
is_running = 1
step=0
reach_step = 50 # number of steps to reach each target
div_step = 10 # number of steps divided for each goal trajectory
train_set=[]

data_file = open("data1.npy","wb")  # one sample in file, with number of steps = 2*div_step

# use [:] to prevent copying pointer instead of copying the array, 
# A=B (array), if A is changed with like += ,etc operations, will cause B to be changed!
# use [:] will prevent B being changed with A's change, A is a copy of array B
joint_angles=start_joint_angles[:]  

sample_batch=0
target_pos = INTER_GOAL  # initial goal

while True:
    step+=1

    # reach the intermediate goal 
    if step  == reach_step:
        print('Reach intermediate goal: ', target_pos)
        target_joint_angles=joint_angles
        # print(target_joint_angles,start_joint_angles)
        action=(np.array(target_joint_angles)-np.array(start_joint_angles))/div_step
        state=get_state(start_joint_angles,target_pos)
        step_joint_angles=start_joint_angles
        for i in range (div_step):
            train_set.append([state, action])
            step_joint_angles = step_joint_angles + action
            state=get_state(step_joint_angles,target_pos)

            ''' display the trajectory of data samples in real (x-y) space'''
            pygame.draw.circle(screen, (0, 0, 255), [state[6],state[7]], 3)
            pygame.display.flip()
            time.sleep(0.1)

        target_pos = TARGET_GOAL  # final goal
        inter_joint_angles = target_joint_angles[:]


    # reach the final goal 
    if step  == 2*reach_step:
        print('Reach target goal: ', target_pos)
        start_joint_angles=inter_joint_angles[:]
        target_joint_angles=joint_angles[:]
        # print(target_joint_angles,start_joint_angles)
        action=(np.array(target_joint_angles)-np.array(start_joint_angles))/div_step
        state=get_state(start_joint_angles,target_pos)
        step_joint_angles=start_joint_angles[:]
        for i in range (div_step):
            train_set.append([state, action])
            step_joint_angles = step_joint_angles + action
            state=get_state(step_joint_angles,target_pos)

            ''' display the trajectory of data samples in real (x-y) space'''
            pygame.draw.circle(screen, (0, 0, 255), [state[6],state[7]], 3)
            pygame.display.flip()
            time.sleep(0.1)

        # save sample
        print('data dim: ', len(train_set))
        print(train_set)
        # pickle.dump( train_set, data_file )
        np.save(data_file, train_set)
        data_file.close()
        train_set=[]
        break
        

    
    # Compute the difference between the target pose and the robot's end pose
    end_pose = compute_end_pose([joint_angles[0], joint_angles[1], joint_angles[2]])
    end_pose_delta = target_pos - end_pose
    # Compute the Jacobian at the current configuration
    ''' an example of joint_angles with 2 small value will cause ill large value for d_theta'''
    # joint_angles=[1, -1e-5,2e-5]
    jacobian = compute_jacobian(joint_angles)
    # Compute the pseudo-inverse of this Jacobian
    jacobian_inverse = np.linalg.pinv(jacobian)
    # Compute the change in joint angles
    d_theta = np.dot(jacobian_inverse, end_pose_delta)
    # Compute the new joint angles
    print('angles: ',joint_angles)
    # print('step: {} d_theta: {}'.format(step, d_theta))
    joint_angles[0] = joint_angles[0] + step_size * d_theta[0]
    joint_angles[1] = joint_angles[1] + step_size * d_theta[1]
    joint_angles[2] = joint_angles[2] + step_size * d_theta[2]

    # Draw the robot in its new state

    state_new=draw_current_state(joint_angles, target_pos)


data_file.close()

