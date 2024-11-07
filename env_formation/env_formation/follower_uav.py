import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import random
import math
import os
from torch.distributions import Normal
import collections
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append(os.path.dirname(__file__))
# from env_formation.masac import ReplayBuffer, PolicyNetwork




class follower_uav():
    def __init__(self, radius=5, pos=[25, 25], vel=[0,0], memo_size=100000, state_dim=40, action_dim=2, alpha=1e-4, beta=1e-4,
                 alpha_lr=1e-4, hidden_dim=600, gamma=0.99, tau=0.01, batch_size=512, target_entropy=-math.log(2)):
        self.radius = radius
        self.pos = pos
        self.vel = vel
        self.memo_size = memo_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.beta = beta
        self.alpha_lr = alpha_lr
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.target_entropy = target_entropy
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.obs_done = False
        self.side_done = False
        self.uav_done = False
        self.done = False
        self.target = False
        self.observation = np.array([])

    def set_position(self, x, y):
        self.pos[0] = x
        self.pos[1] = y

    def set_vel(self, x, y):
        self.vel[0] = x
        self.vel[1] = y
    
    def position(self):
        return np.array(self.pos)