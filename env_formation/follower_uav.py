import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import random
import math
import os
from torch.distributions import Normal
import collections

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        transations = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transations)
        return(
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done)
        )
    
    def size(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)

        log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        log_prob = log_prob.sum(dim = -1, keepdim=True)

        return action, log_prob
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkoint(self, checkpont_file):
        self.load_state_dict(torch.load(checkpont_file))


    

class QvalueNet(nn.Module):
    def __init__(self, multi_state_dim, multi_hidden_dim, multi_action_dim):
        super(QvalueNet, self).__init__()
        self.fc1 = nn.Linear(multi_state_dim + multi_action_dim, multi_hidden_dim)
        self.fc2 = nn.Linear(multi_hidden_dim, multi_hidden_dim)
        self.fc_out = nn.Linear(multi_hidden_dim, 1)
    
    def forward(self, mx, ma):
        cat = torch.cat([mx, ma], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        out_put = self.fc_out(x)

        return out_put
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))


class SAC:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device, agent_num):
        self.actor = PolicyNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1 = QvalueNet(state_dim*agent_num, hidden_dim*agent_num, action_dim).to(device)    
        self.critic_2 = QvalueNet(state_dim*agent_num, hidden_dim*agent_num, action_dim).to(device)    
        self.target_critic_1 = QvalueNet(state_dim*agent_num, hidden_dim*agent_num, action_dim).to(device)    
        self.target_critic_2 = QvalueNet(state_dim*agent_num, hidden_dim*agent_num, action_dim).to(device)    
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad=True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy  =target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.state_dim = state_dim

    def take_action(self, state):
        state = torch.tensor(np.array(state),dtype=torch.float).to(self.device)
        action = self.actor(state)[0]
        action = action.cpu().detach().numpy().flatten()
        return action
    
    def calc_target(self, rewards, next_states, multi_next_states, dones):
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(multi_next_states, next_actions)
        q2_value = self.target_critic_2(multi_next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy

        td_target = rewards + self.gamma * next_value * (1-dones)

        return td_target
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        multi_state = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        multi_next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = self.calc_target(rewards=rewards, next_states=multi_next_states[:self.state_dim], multi_next_states=multi_next_states, dones=dones)#TODO

        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(multi_state, actions), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(multi_state, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        new_actions, log_prob = self.actor(multi_state[:self.state_dim])#TODO
        entropy = -log_prob
        q1_value = self.critic_1(multi_state, new_actions)
        q2_value = self.critic_2(multi_state, new_actions)

        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss=torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def save_model(self, base_path, scenario, id):
        self.actor.save_checkpoint(os.path.join(base_path, f"uav_{id}_actor_{scenario}.pth"))
        self.critic_1.save_checkpoint(os.path.join(base_path, f"uav_{id}_critic_1_{scenario}.pth"))
        self.critic_2.save_checkpoint(os.path.join(base_path, f"uav_{id}_critic_2_{scenario}.pth"))
        self.target_critic_1.save_checkpoint(os.path.join(base_path, f"uav_{id}_target_critic_1_{scenario}.pth"))
        self.target_critic_2.save_checkpoint(os.path.join(base_path, f"uav_{id}_target_critic_2_{scenario}.pth"))

    def load_model(self, base_path, scenario, id):
        self.actor.load_checkoint(os.path.join(base_path, f"uav_{id}_actor_{scenario}.pth"))
        self.critic_1.load_checkpoint(os.path.join(base_path, f"uav_{id}_critic_1_{scenario}.pth"))
        self.critic_2.load_checkpoint(os.path.join(base_path, f"uav_{id}_critic_2_{scenario}.pth"))
        self.target_critic_1.load_checkpoint(os.path.join(base_path, f"uav_{id}_target_critic_1_{scenario}.pth"))
        self.target_critic_2.load_checkpoint(os.path.join(base_path, f"uav_{id}_target_critic_2_{scenario}.pth"))

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
        self.done = False
        self.target = False
        self.observation = np.array([])

        self.sac_network = SAC(state_dim=self.state_dim, hidden_dim=self.hidden_dim, action_dim=self.action_dim,
                               actor_lr=self.alpha, critic_lr=self.beta, alpha_lr=self.alpha_lr,
                               target_entropy=self.target_entropy, tau=self.tau, gamma=self.gamma,
                               device=self.device)
        self.replay_buffer = ReplayBuffer(capacity=self.memo_size)

    def set_position(self, x, y):
        self.pos[0] = x
        self.pos[1] = y

    def set_vel(self, x, y):
        self.vel[0] = x
        self.vel[1] = y
    
    def position(self):
        return np.array([self.pos])