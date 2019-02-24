import pygame
import numpy as np
import math

# Global variables
screen_size = 1000
link_lengths = [200, 140, 100, 80]
joint_angles = [0, 0, 0, 0]


# Function to compute the transformation matrix between two frames
def compute_trans_mat(angle, length):
    cos_theta = math.cos(math.radians(angle))
    sin_theta = math.sin(math.radians(angle))
    dx = -length * sin_theta
    dy = length * cos_theta
    T = np.array([[cos_theta, -sin_theta, dx], [sin_theta, cos_theta, dy], [0, 0, 1]])
    return T


# Function to draw the current state of the world
def draw_current_state():
    # First link in world coordinates
    T_01 = compute_trans_mat(joint_angles[0], link_lengths[0])
    origin_1 = np.dot(T_01, np.array([0, 0, 1]))
    p0 = [0, 0]
    p1 = [origin_1[0], -origin_1[1]]  # the - is because the y-axis is opposite in world and image coordinates
    # Second link in world coordinates
    T_12 = compute_trans_mat(joint_angles[1], link_lengths[1])
    origin_2 = np.dot(T_01, np.dot(T_12, np.array([0, 0, 1])))
    p2 = [origin_2[0], -origin_2[1]]  # the - is because the y-axis is opposite in world and image coordinates
    # Third link in world coordinates
    T_23 = compute_trans_mat(joint_angles[2], link_lengths[2])
    origin_3 = np.dot(T_01, np.dot(T_12, np.dot(T_23, np.array([0, 0, 1]))))
    p3 = [origin_3[0], -origin_3[1]]  # the - is because the y-axis is opposite in world and image coordinates
    # Compute the screen coordinates
    p0_u = int(0.5 * screen_size + p0[0])
    p0_v = int(0.5 * screen_size + p0[1])
    p1_u = int(0.5 * screen_size + p1[0])
    p1_v = int(0.5 * screen_size + p1[1])
    p2_u = int(0.5 * screen_size + p2[0])
    p2_v = int(0.5 * screen_size + p2[1])
    p3_u = int(0.5 * screen_size + p3[0])
    p3_v = int(0.5 * screen_size + p3[1])
    # Draw
    screen.fill((0, 0, 0))
    pygame.draw.line(screen, (255, 255, 255), [p0_u, p0_v], [p1_u, p1_v], 5)
    pygame.draw.line(screen, (255, 255, 255), [p1_u, p1_v], [p2_u, p2_v], 5)
    pygame.draw.line(screen, (255, 255, 255), [p2_u, p2_v], [p3_u, p3_v], 5)
    pygame.draw.circle(screen, (0, 255, 0), [p0_u, p0_v], 10)
    pygame.draw.circle(screen, (0, 0, 255), [p1_u, p1_v], 10)
    pygame.draw.circle(screen, (0, 0, 255), [p2_u, p2_v], 10)
    pygame.draw.circle(screen, (255, 0, 0), [p3_u, p3_v], 10)
    # Flip the display buffers to show the current rendering
    pygame.display.flip()


# The main entry point
screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption("Reacher")
is_running = 1
# Loop until the window is closed
while is_running:
    # Get events and check if the user has closed the window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = 0
            break
    # Change the joint angles (the increment is in degrees)
    joint_angles[0] += 0.1
    joint_angles[1] += 0.2
    joint_angles[2] += 18
    # Draw the robot in its new state
    draw_current_state()
