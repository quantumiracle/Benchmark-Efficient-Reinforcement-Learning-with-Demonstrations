import pygame
import numpy as np
import math
import time
from gym.spaces.box import Box

class Reacher:
    def __init__(self, screen_size=1000, link_lengths = [200, 140, 100], joint_angles=[0, 0, 0], target_pos = [619,330], render=False):
        # Global variables
        self.screen_size = screen_size
        self.link_lengths = link_lengths
        self.joint_angles = joint_angles
        self.num_actions=3  # equals to number of joints - 1
        self.num_observations= 2*(self.num_actions+2)
        self.L = 8 # distance from target to get reward 2
        self.action_space=Box(-100,100, [self.num_actions])
        self.observation_space=Box(-1000,1000, [2*(self.num_actions+2)])

        # The main entry point
        self.render=render
        if self.render == True:
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Reacher")
        else:
            pass
        self.is_running = 1
        # self.target_pos=[self.screen_size/4, self.screen_size/4]
        self.target_pos=target_pos

        self.steps=0
        self.max_episode_steps=50
        self.near_goal_range=0.5


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
        # print(p0,p1,p2,p3)
        p0_u = int(0.5 * self.screen_size + p0[0])
        p0_v = int(0.5 * self.screen_size + p0[1])
        p1_u = int(0.5 * self.screen_size + p1[0])
        p1_v = int(0.5 * self.screen_size + p1[1])
        p2_u = int(0.5 * self.screen_size + p2[0])
        p2_v = int(0.5 * self.screen_size + p2[1])
        p3_u = int(0.5 * self.screen_size + p3[0])
        p3_v = int(0.5 * self.screen_size + p3[1])
        # Draw
        if self.render == True:
            self.screen.fill((0, 0, 0))
            pygame.draw.line(self.screen, (255, 255, 255), [p0_u, p0_v], [p1_u, p1_v], 5)
            pygame.draw.line(self.screen, (255, 255, 255), [p1_u, p1_v], [p2_u, p2_v], 5)
            pygame.draw.line(self.screen, (255, 255, 255), [p2_u, p2_v], [p3_u, p3_v], 5)
            pygame.draw.circle(self.screen, (0, 255, 0), [p0_u, p0_v], 10)
            pygame.draw.circle(self.screen, (0, 0, 255), [p1_u, p1_v], 10)
            pygame.draw.circle(self.screen, (0, 0, 255), [p2_u, p2_v], 10)
            pygame.draw.circle(self.screen, (255, 0, 0), [p3_u, p3_v], 10)
            
            pygame.draw.circle(self.screen, (255, 255, 0), np.array(self.target_pos).astype(int), 10)
            # Flip the display buffers to show the current rendering
            pygame.display.flip()
        else:
            pass
        return [p0_u,p0_v,p1_u,p1_v,p2_u,p2_v,p3_u,p3_v]
    
    def reset(self,):
        self.steps=0
        self.joint_angles = np.array([0.1,0.1,0.1])*180.0/np.pi
        if self.render == True:
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Reacher")
        else:
            pass
        self.is_running = 1
        pos_set=self.draw_current_state()
        # return np.array([np.concatenate((pos_set,self.link_lengths))])
        return np.array([np.concatenate((pos_set,self.target_pos))]).reshape(-1)

    def step(self,action):    
        # Get events and check if the user has closed the window
        if self.render == True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = 0
                    break
        else:
            pass
        # Change the joint angles (the increment is in degrees)
        change=np.random.uniform(-1,1,size=3)
        # self.joint_angles[0] += 0.1
        # self.joint_angles[1] += 0.2
        # self.joint_angles[2] += 0.3
        # self.joint_angles[0] += change[0]
        # self.joint_angles[1] += change[1]
        # self.joint_angles[2] += change[2]
        # print(action)
        self.joint_angles[0] += action[0]
        self.joint_angles[1] += action[1]
        self.joint_angles[2] += action[2]
        # Draw the robot in its new state
        # print(action)
        pos_set=self.draw_current_state()
        # if abs(pos_set[6]-self.target_pos[0])<self.L and abs(pos_set[7]-self.target_pos[1])<self.L:
        #     reward = 2
        # else:
        #     reward = 0

        # reward_0=1000
        # reward = reward_0 * np.exp(-np.sqrt(abs(pos_set[6]-self.target_pos[0])**2+abs(pos_set[7]-self.target_pos[1])**2))
        # print(reward) #e-100

        reward_0=100.0
        pos2goal_distance=np.sqrt(abs(pos_set[6]-self.target_pos[0])**2+abs(pos_set[7]-self.target_pos[1])**2)
        reward = reward_0 / (pos2goal_distance+1)
        # time.sleep(0.2)
        self.steps+=1
        if self.steps > self.max_episode_steps:
            self.steps=0
            return np.array([np.concatenate((pos_set,self.target_pos))]).reshape(-1), reward, True
        else:
            return np.array([np.concatenate((pos_set,self.target_pos))]).reshape(-1), reward, False


if __name__ == "__main__":
    screen_size = 1000
    # link_lengths = [200, 140, 100, 80]
    link_lengths = [200, 140, 100]
    joint_angles = [0, 0, 0, 0]
    target_pos=[screen_size/4, screen_size/4]
    reacher=Reacher(screen_size, link_lengths, joint_angles,target_pos)
    # reacher.reset()
    num_steps=50
    # Loop until the window is closed
    step=0
    while reacher.is_running:
        print(step)
        step+=1
        reacher.step()
        if step >= num_steps:
            reacher.is_running=0
    # for step in range (num_steps):
    #     print(step)
    #     if reacher.is_running:
    #         reacher.step()
    

    reacher.reset()
    # print(reacher.is_running)
    step=0
    while reacher.is_running:
        print(step)
        step+=1
        pos=reacher.step()
        print(pos,len(pos))
        if step >= num_steps:
            reacher.is_running=0


    
