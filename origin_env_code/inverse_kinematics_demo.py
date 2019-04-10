# Python imports
import pygame
import numpy as np
import numdifftools as nd


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
    pygame.display.flip()


# The main entry point
# Define some parameters
screen_size = 1000  # The size of the rendered screen in pixels
link_lengths = [0.2, 0.15, 0.1]  # These lengths are in world space (0 to 1), not screen space (0 to 1000)
step_size = 0.1  # This is the gradient descent step taken when solving the inverse kinematics
# Set the initial joint angles and target pose (all angles are in radians throughout program)
target_pose = [0.3, 0.3]
joint_angles = [0.1, 0.1, 0.1]
# Create a PyGame instance
screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption("Inverse Kinematics Demo")
is_running = 1
# Main program loop
while is_running:
    # Check for mouse events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = 0
            break
        elif event.type == pygame.MOUSEBUTTONUP:
            # Set a new target
            target_screen = pygame.mouse.get_pos()
            target_pose = [(target_screen[0] / screen_size) - 0.5, -((target_screen[1] / screen_size) - 0.5)]
    # Compute the difference between the target pose and the robot's end pose
    end_pose = compute_end_pose([joint_angles[0], joint_angles[1], joint_angles[2]])
    end_pose_delta = target_pose - end_pose
    # Compute the Jacobian at the current configuration
    jacobian = compute_jacobian(joint_angles)
    # Compute the pseudo-inverse of this Jacobian
    jacobian_inverse = np.linalg.pinv(jacobian)
    # Compute the change in joint angles
    d_theta = np.dot(jacobian_inverse, end_pose_delta)
    # Compute the new joint angles
    joint_angles[0] += step_size * d_theta[0]
    joint_angles[1] += step_size * d_theta[1]
    joint_angles[2] += step_size * d_theta[2]
    # Draw the robot in its new state
    draw_current_state(joint_angles, target_pose)
