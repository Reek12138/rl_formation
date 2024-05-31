import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Critic(nn.Module):
    def __init__(self, lr_critic, input_dims, fc1_dims, fc2_dims, n_agent,action_dim) :
        super(Critic, self).__init__()

        self.fcl = nn.Linear(input_dims, fc1_dims)
        self.fc2_s = nn.Linear(fc1_dims, fc2_dims)
        self.fc2_a = nn.Linear(action_dim, fc2_dims)
        self.q = nn.Linear(fc2_dims, out_features=1)

        self.fcl_ = nn.Linear(input_dims, fc1_dims)
        self.fc2_s_ = nn.Linear(fc1_dims, fc2_dims)
        self.fc2_a_ = nn.Linear(action_dim, fc2_dims)
        self.q_ = nn.Linear(fc2_dims, out_features=1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_critic)

    def forward(self, state,action):
        # x = torch.cat([state, action], dim=1)
        # x = F.relu((self.fcl(x)))
        # x = F.relu((self.fc2(x)))
        # q = self.q(x)
        # return q
        s1 = F.relu(self.fcl(state))
        self.fc2_s(s1)
        self.fc2_a(action)
        s11 = torch.mm(s1, self.fc2_s.weight.data.t())
        s12 = torch.mm(action, self.fc2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.fc2_a.bias.data)
        q1 = self.q(s1)

        s2 = F.relu(self.fcl_(state))
        self.fc2_s_(s2)
        self.fc2_a_(action)
        s21 = torch.mm(s2, self.fc2_s_.weight.data.t())
        s22 = torch.mm(action, self.fc2_a_.weight.data.t())
        s2 = F.relu(s21 + s22 + self.fc2_a_.bias.data)
        q2 = self.q(s2)

        return q1, q2
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkppint_file):
        self.load_state_dict(torch.load(checkppint_file))


class Actor(nn.Module):
    def __init__(self, lr_actor, input_dims, fc1_dims, fc2_dims, action_dim):
        super(Actor, self).__init__()

        self.fcl = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, action_dim)
        self.tanh = nn.Tanh()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_actor)

    def forward(self, state):
        x = F.relu((self.fcl(state)))
        x = F.relu((self.fc2(x)))
        # mu = torch.softmax(self.pi(x),dim=1)
        mu = self.tanh(self.pi(x))
        return mu
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkppint_file):
        self.load_state_dict(torch.load(checkppint_file))

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, state_dim, action_dim, batch_size):
        self.capacity = capacity
        self.obs_cap = np.empty((self.capacity, obs_dim))
        self.next_obs_cap = np.empty((self.capacity, obs_dim))
        self.state_cap =  np.empty((self.capacity, state_dim))
        self.next_state_cap =  np.empty((self.capacity, state_dim))
        self.action_cap = np.empty((self.capacity, action_dim))
        self.reward_cap = np.empty((self.capacity, 1))
        self.done_cap = np.empty((self.capacity, 1),dtype=bool)
        
        self.batch_size = batch_size
        self.current = 0

    def add_memo(self, obs, next_obs, state, next_state, action, reward, done):
        
        self.obs_cap[self.current] = obs
        self.next_obs_cap[self.current] = next_obs
        self.state_cap[self.current] = state
        self.next_state_cap[self.current] = next_state
        self.action_cap[self.current] = action
        self.reward_cap[self.current] = reward
        self.done_cap[self.current] = done

        self.current = (self.current + 1) % self.capacity

    def sample(self, indxes):
        obs = self.obs_cap[indxes]
        next_obs = self.next_obs_cap[indxes]
        state = self.state_cap[indxes]
        next_state = self.next_state_cap[indxes]
        action = self.action_cap[indxes]
        reward = self.reward_cap[indxes]
        done = self.done_cap[indxes]
        
        return obs, next_obs, state, next_state, action, reward, done
    
    
class circle_agent():
    def __init__(self, radius, pos_x, pos_y, linear_vel = 0, orientation_vel = 0, orientation = np.pi/4, vel_x=0, vel_y=0,
                memo_size = 100000, obs_dim = 40, state_dim = 5, n_agent = 4, action_dim = 2, alpha = 0.01 , beta = 0.01, 
                fc1_dims = 640,fc2_dims = 640, gamma = 0.9 , tau = 0.01, batch_size = 512, max_vel = 10) -> None:
        
        #智能体的信息
        self.radius = radius
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.observation = {}
        self.reward = 0.0
        self.done = False
        self.target = False
        self.info = {}
        self.xy_vel = [vel_x, vel_y]
        self.linear_orientation = [linear_vel,  orientation_vel]
        self.orientation = orientation
        self.max_vel = max_vel

        #智能体rl配置
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        self.actor = Actor(lr_actor=alpha, input_dims=obs_dim, fc1_dims=fc1_dims, 
                           fc2_dims=fc2_dims, action_dim=action_dim).to(device)

        self.critic = Critic(lr_critic=beta, input_dims=state_dim, fc1_dims=fc1_dims, 
                             fc2_dims=fc2_dims, n_agent=n_agent, action_dim=action_dim).to(device)
        
        self.target_actor = Actor(lr_actor=alpha, input_dims=obs_dim, fc1_dims=fc1_dims, 
                           fc2_dims=fc2_dims, action_dim=action_dim).to(device)

        self.target_critic = Critic(lr_critic=beta, input_dims=state_dim, fc1_dims=fc1_dims, 
                             fc2_dims=fc2_dims, n_agent=n_agent, action_dim=action_dim).to(device)
        
        self.replay_buffer = ReplayBuffer(capacity = memo_size, obs_dim = obs_dim, state_dim = state_dim, action_dim = action_dim, batch_size = batch_size)
    

    def set_position(self, x, y):
        """设置代理的当前位置"""
        self.pos_x = x
        self.pos_y = y
    def position(self):
        return np.array([self.pos_x, self.pos_y])
    
    def set_xy_vel(self, vx, vy):
        self.vel = [vx, vy]

    def set_linear_orientation(self, linear_vel, orientation):
        self.linear_orientation = [linear_vel, orientation]
    
    def get_action(self, obs, mode):
        single_obs = torch.tensor(data=obs, dtype=torch.float).unsqueeze(0).to(device)
        single_action = self.actor.forward(single_obs)
        noise = torch.randn(self.action_dim).to(device)
        # single_action = torch.clamp(input=single_action + noise, min=0.0, max=1.0)
        # Combine action and noise in 1:0.5 ratio
        

        if mode == "a":
            mixed_action = single_action + noise*0.8
        elif mode == "b":
            mixed_action = single_action + noise*0.6
        elif mode == "c":
            mixed_action = single_action + noise*0.4
        elif mode == "d":
            mixed_action = single_action + noise*0.2
        
        # Clamp the mixed action to be within valid bounds
        clamped_action = torch.clamp(mixed_action, min=-1.0, max=1.0)
        action = clamped_action.detach().cpu().numpy()[0]
        
        return action
    
    def save_model(self, filename):
        self.actor.save_checkpoint(filename)
        self.target_actor.save_checkpoint(filename)
        self.critic.save_checkpoint(filename)
        self.target_critic.save_checkpoint(filename)

    def load_model(self, filename):
        self.actor.load_checkpoint(filename)
        self.target_actor.load_checkpoint(filename)
        self.critic.load_checkpoint(filename)
        self.target_critic.load_checkpoint(filename)