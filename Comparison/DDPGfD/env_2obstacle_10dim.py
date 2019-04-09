'''
env with an obstacle area with negative reward value
'''

import pygame
import numpy as np
import math
import time




class Reacher:
    def __init__(self, screen_size, link_lengths, joint_angles):
        # Global variables
        self.screen_size = screen_size
        self.link_lengths = link_lengths
        self.joint_angles = joint_angles
        self.num_actions=3  # equals to number of joints - 1
        self.L = 30 # distance from target to get reward 2

        # The main entry point
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Reacher")
        self.is_running = 1
        self.ini_pos=[480, 60]
        self.target_pos=[self.screen_size/4, self.screen_size*3/4]
        
        self.OBSTACLE_RADIUS = 50
        self.OBSTACLE_PANELTY = -5
        self.OBSTACLE_DISTANCE = 280
        self.NUM_OBSTACLES = 2
        self.obstacle1_pos=0.5 * ( np.array(self.ini_pos)+np.array(self.target_pos))
        self.obstacle2_pos=0.5 * ( np.array(self.ini_pos)+np.array(self.target_pos)) - np.array([self.OBSTACLE_DISTANCE,0])
        print(self.obstacle1_pos, self.obstacle2_pos)

    # Function to compute the transformation matrix between two frames
    def compute_trans_mat(self, angle, length):
        cos_theta = math.cos(math.radians(angle))
        sin_theta = math.sin(math.radians(angle))
        dx = -length * sin_theta
        dy = length * cos_theta
        T = np.array([[cos_theta, -sin_theta, dx], [sin_theta, cos_theta, dy], [0, 0, 1]])
        return T


    # Function to draw the current state of the world
    def draw_current_state(self, ):
        # First link in world coordinates
        T_01 = self.compute_trans_mat(self.joint_angles[0], self.link_lengths[0])
        origin_1 = np.dot(T_01, np.array([0, 0, 1]))
        p0 = [0, 0]
        p1 = [origin_1[0], -origin_1[1]]  # the - is because the y-axis is opposite in world and image coordinates
        # Second link in world coordinates
        T_12 = self.compute_trans_mat(self.joint_angles[1], self.link_lengths[1])
        origin_2 = np.dot(T_01, np.dot(T_12, np.array([0, 0, 1])))
        p2 = [origin_2[0], -origin_2[1]]  # the - is because the y-axis is opposite in world and image coordinates
        # Third link in world coordinates
        T_23 = self.compute_trans_mat(self.joint_angles[2], self.link_lengths[2])
        origin_3 = np.dot(T_01, np.dot(T_12, np.dot(T_23, np.array([0, 0, 1]))))
        p3 = [origin_3[0], -origin_3[1]]  # the - is because the y-axis is opposite in world and image coordinates
        # Compute the screen coordinates
        p0_u = int(0.5 * self.screen_size + p0[0])
        p0_v = int(0.5 * self.screen_size + p0[1])
        p1_u = int(0.5 * self.screen_size + p1[0])
        p1_v = int(0.5 * self.screen_size + p1[1])
        p2_u = int(0.5 * self.screen_size + p2[0])
        p2_v = int(0.5 * self.screen_size + p2[1])
        p3_u = int(0.5 * self.screen_size + p3[0])
        p3_v = int(0.5 * self.screen_size + p3[1])
        # Draw
        self.screen.fill((0, 0, 0))
        pygame.draw.line(self.screen, (255, 255, 255), [p0_u, p0_v], [p1_u, p1_v], 5)
        pygame.draw.line(self.screen, (255, 255, 255), [p1_u, p1_v], [p2_u, p2_v], 5)
        pygame.draw.line(self.screen, (255, 255, 255), [p2_u, p2_v], [p3_u, p3_v], 5)
        pygame.draw.circle(self.screen, (0, 255, 0), [p0_u, p0_v], 10)
        pygame.draw.circle(self.screen, (0, 0, 255), [p1_u, p1_v], 10)
        pygame.draw.circle(self.screen, (0, 0, 255), [p2_u, p2_v], 10)
        pygame.draw.circle(self.screen, (255, 0, 0), [p3_u, p3_v], 10)
        
        pygame.draw.circle(self.screen, (255, 255, 0), np.array(self.target_pos).astype(int), 10)
        pygame.draw.circle(self.screen, (125, 125, 0), np.array(self.obstacle1_pos).astype(int), self.OBSTACLE_RADIUS)
        pygame.draw.circle(self.screen, (125, 125, 0), np.array(self.obstacle2_pos).astype(int), self.OBSTACLE_RADIUS)
        # Flip the display buffers to show the current rendering
        pygame.display.flip()
        return [p0_u,p0_v,p1_u,p1_v,p2_u,p2_v,p3_u,p3_v]
    
    def reset(self,):
        self.joint_angles = [0.1, 0.1, 0.1]
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Reacher")
        self.is_running = 1
        pos_set=self.draw_current_state()
        return np.array([np.concatenate((pos_set,self.target_pos))])

    def step(self,action, sparse_reward):    
        # Get events and check if the user has closed the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = 0
                break
        # Change the joint angles (the increment is in degrees)
        change=np.random.uniform(-1,1,size=3)
        # self.joint_angles[0] += 0.1
        # self.joint_angles[1] += 0.2
        # self.joint_angles[2] += 0.3
        # self.joint_angles[0] += change[0]
        # self.joint_angles[1] += change[1]
        # self.joint_angles[2] += change[2]
        # print(action)
        self.joint_angles[0] += action[0][0]
        self.joint_angles[1] += action[0][1]
        self.joint_angles[2] += action[0][2]
        # Draw the robot in its new state
        pos_set=self.draw_current_state()
        if sparse_reward:
            '''sparse reward'''
            distance2goal = np.sqrt((pos_set[6]-self.target_pos[0])**2+(pos_set[7]-self.target_pos[1])**2)
            distance2obstacle1 = np.sqrt((pos_set[6]-self.obstacle1_pos[0])**2+(pos_set[7]-self.obstacle1_pos[1])**2)
            distance2obstacle2 = np.sqrt((pos_set[6]-self.obstacle2_pos[0])**2+(pos_set[7]-self.obstacle2_pos[1])**2)
            if distance2goal < self.L:
                reward = 20
            elif distance2obstacle1 < self.OBSTACLE_RADIUS or distance2obstacle2 < self.OBSTACLE_RADIUS:
                reward = self.OBSTACLE_PANELTY
            else:
                reward = -1
            return np.array([np.concatenate((pos_set,self.target_pos))]), np.array([reward]), np.array([False]), distance2goal
        
        else:    
            '''dense reward'''
            reward_0=100.0
            reward = reward_0 / (np.sqrt((pos_set[6]-self.target_pos[0])**2+(pos_set[7]-self.target_pos[1])**2)+1)
            if np.sqrt((pos_set[6]-self.obstacle1_pos[0])**2+(pos_set[7]-self.obstacle1_pos[1])**2) < self.OBSTACLE_RADIUS:
                reward += self.OBSTACLE_PANELTY
                print('-5!')
            if np.sqrt((pos_set[6]-self.obstacle2_pos[0])**2+(pos_set[7]-self.obstacle2_pos[1])**2) < self.OBSTACLE_RADIUS:
                reward += self.OBSTACLE_PANELTY
                print('-5!')

            # time.sleep(0.3)
            # 10 dim return
            return np.array([np.concatenate((pos_set,self.target_pos))]), np.array([reward]), np.array([False])


if __name__ == "__main__":
    screen_size = 1000
    # link_lengths = [200, 140, 100, 80]
    link_lengths = [200, 140, 100]
    joint_angles = [0, 0, 0, 0]
    reacher=Reacher(screen_size, link_lengths, joint_angles)
    # reacher.reset()
    num_steps=50
    # Loop until the window is closed
    step=0
    while reacher.is_running:
        print(step)
        action=np.random.rand(1,3)
        step+=1
        time.sleep(0.5)
        reacher.step(action)
        if step >= num_steps:
            reacher.is_running=0
    # for step in range (num_steps):
    #     print(step)
    #     if reacher.is_running:
    #         reacher.step()
    

    reacher.reset()


    
