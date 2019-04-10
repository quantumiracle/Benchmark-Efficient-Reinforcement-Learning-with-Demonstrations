import pygame
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class Reacher:
    def __init__(self, screen_size, link_lengths, joint_angles):
        # Global variables
        self.screen_size = screen_size
        self.link_lengths = link_lengths
        self.joint_angles = joint_angles
        self.num_actions=5  # equals to number of joints - 1
        self.L = 8 # distance from target to get reward 2

        # The main entry point
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Reacher")
        self.is_running = 1
        self.target_pos=[self.screen_size*7/8, self.screen_size/4]
        self.penalty_pos1=[self.screen_size*6/8, self.screen_size/4]
        self.penalty_pos2=[self.screen_size*7/8, self.screen_size/8]


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
        
        T_34 = self.compute_trans_mat(self.joint_angles[3], self.link_lengths[3])
        origin_4 = np.dot(T_01, np.dot(T_12, np.dot(T_23, np.dot(T_34, np.array([0, 0, 1])))))
        p4 = [origin_4[0], -origin_4[1]]

        T_45 = self.compute_trans_mat(self.joint_angles[4], self.link_lengths[4])
        origin_5 = np.dot(T_01, np.dot(T_12, np.dot(T_23, np.dot(T_34, np.dot(T_45, np.array([0, 0, 1]))))))
        p5 = [origin_5[0], -origin_5[1]]
        
        # Compute the screen coordinates
        p0_u = int(0.5 * self.screen_size + p0[0])
        p0_v = int(0.5 * self.screen_size + p0[1])
        p1_u = int(0.5 * self.screen_size + p1[0])
        p1_v = int(0.5 * self.screen_size + p1[1])
        p2_u = int(0.5 * self.screen_size + p2[0])
        p2_v = int(0.5 * self.screen_size + p2[1])
        p3_u = int(0.5 * self.screen_size + p3[0])
        p3_v = int(0.5 * self.screen_size + p3[1])
        p4_u = int(0.5 * self.screen_size + p4[0])
        p4_v = int(0.5 * self.screen_size + p4[1])
        p5_u = int(0.5 * self.screen_size + p5[0])
        p5_v = int(0.5 * self.screen_size + p5[1])
        # Draw
        self.screen.fill((0, 0, 0))
        pygame.draw.line(self.screen, (255, 255, 255), [p0_u, p0_v], [p1_u, p1_v], 5)
        pygame.draw.line(self.screen, (255, 255, 255), [p1_u, p1_v], [p2_u, p2_v], 5)
        pygame.draw.line(self.screen, (255, 255, 255), [p2_u, p2_v], [p3_u, p3_v], 5)
        pygame.draw.line(self.screen, (255, 255, 255), [p3_u, p3_v], [p4_u, p4_v], 5)
        pygame.draw.line(self.screen, (255, 255, 255), [p4_u, p4_v], [p5_u, p5_v], 5)

        pygame.draw.circle(self.screen, (0, 255, 0), [p0_u, p0_v], 10)
        pygame.draw.circle(self.screen, (0, 0, 255), [p1_u, p1_v], 10)
        pygame.draw.circle(self.screen, (0, 0, 255), [p2_u, p2_v], 10)
        pygame.draw.circle(self.screen, (255, 0, 0), [p3_u, p3_v], 10)
        pygame.draw.circle(self.screen, (255, 125, 0), [p4_u, p4_v], 10)
        pygame.draw.circle(self.screen, (255, 0, 125), [p5_u, p5_v], 10)
        
        pygame.draw.circle(self.screen, (255, 255, 0), np.array(self.target_pos).astype(int), 10)
        pygame.draw.circle(self.screen, (100, 100, 0), np.array(self.penalty_pos1).astype(int), 10)
        pygame.draw.circle(self.screen, (100, 100, 0), np.array(self.penalty_pos2).astype(int), 10)

        # Flip the display buffers to show the current rendering
        pygame.display.flip()
        return [p0_u,p0_v,p1_u,p1_v,p2_u,p2_v,p3_u,p3_v,p4_u,p4_v,p5_u,p5_v]
    
    def reset(self,):
        self.joint_angles = [0, 0, 0, 0,0,0]
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Reacher")
        self.is_running = 1
        pos_set=self.draw_current_state()
        return np.array([np.concatenate((pos_set,self.link_lengths))])

    def step(self,action):    
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
        self.joint_angles[3] += action[0][3]
        self.joint_angles[4] += action[0][4]
        # Draw the robot in its new state
        pos_set=self.draw_current_state()
        # if abs(pos_set[6]-self.target_pos[0])<self.L and abs(pos_set[7]-self.target_pos[1])<self.L:
        #     reward = 2
        # else:
        #     reward = 0

        # reward_0=1000
        # reward = reward_0 * np.exp(-np.sqrt(abs(pos_set[6]-self.target_pos[0])**2+abs(pos_set[7]-self.target_pos[1])**2))
        # print(reward) #e-100

        reward=self.compute_reward(pos_set[10],pos_set[11])
        # time.sleep(0.5)

        return np.array([np.concatenate((pos_set,self.link_lengths))]), np.array([reward]), np.array([False])

    def compute_reward(self,pos_x, pos_y):
        reward_0=10.0
        reward = reward_0 / (np.sqrt(abs(pos_x-self.target_pos[0])**2+abs(pos_y-self.target_pos[1])**2)+1)
        reward = reward - reward_0 / (np.sqrt(abs(pos_x-self.penalty_pos1[0])**2+abs(pos_y-self.penalty_pos1[1])**2)+1)
        reward = reward - reward_0 / (np.sqrt(abs(pos_x-self.penalty_pos2[0])**2+abs(pos_y-self.penalty_pos2[1])**2)+1)
        # ratio=0.01
        # reward = reward_0 / (((abs(pos_x-self.target_pos[0])**2+abs(pos_y-self.target_pos[1])**2)+1)**ratio)
        # reward = reward - reward_0 / (((abs(pos_x-self.penalty_pos1[0])**2+abs(pos_y-self.penalty_pos1[1])**2)+1)**ratio)
        # reward = reward - reward_0 / (((abs(pos_x-self.penalty_pos2[0])**2+abs(pos_y-self.penalty_pos2[1])**2)+1)**ratio)

        # reward = reward / 100.0
        return reward
    
    def visualize_reward(self,  ):
        delta1=1
        # x = np.arange(self.screen_size/2, self.screen_size, delta1)
        # y = np.arange(self.screen_size/16, self.screen_size*3/8, delta1)
        x = np.arange(0, self.screen_size, delta1)
        y = np.arange(0, self.screen_size, delta1)
        X, Y = np.meshgrid(x, y)
        # dx=0.5
        # dy=0.5
        # Y, X = np.mgrid[slice(1, self.screen_size, dy),
        #         slice(1, self.screen_size, dx)]
        Z=self.compute_reward(X,Y)
        # Plot the surface.
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # levels = MaxNLocator(nbins=20).tick_values(Z.min(), Z.max())
        # cmap = plt.get_cmap('PiYG')
        # plt.figure()
        # CS = plt.contourf(X, Y, Z,cmap=cmap, levels=levels)
        fig.gca().invert_yaxis()
        # plt.colorbar(CS)
        # plt.clabel(CS, inline=1, fontsize=10)
        # plt.title('Reward Map')
        # plt.savefig('map.png')
        plt.show()

if __name__ == "__main__":
    screen_size = 1000
    # link_lengths = [200, 140, 100, 80]
    link_lengths = [200, 140, 100,80,60]
    joint_angles = [0, 0, 0, 0,0,0]
    reacher=Reacher(screen_size, link_lengths, joint_angles)
    # reacher.reset()
    num_steps=50
    # Loop until the window is closed
    step=0
    reacher.visualize_reward()
    while reacher.is_running:
        action=np.random.rand(1,5)
        # print(action[0][4])
        print(step)
        step+=1
        reacher.step(action)
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
        action=np.random.rand(1,5)
        print(step)
        step+=1
        pos=reacher.step(action)
        print(pos,len(pos))
        if step >= num_steps:
            reacher.is_running=0


    
