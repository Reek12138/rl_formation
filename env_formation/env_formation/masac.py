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
        std = F.softplus(self.fc_std(x)) + 1e-6
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
        mx = mx.view(mx.size(0), -1)  # 展平为 [batch_size, state_dim * agent_num]
        ma = ma.view(ma.size(0), -1)  # 展平为 [batch_size, action_dim * agent_num]
        # print("Shape of mx:", mx.shape)
        # print("Shape of ma:", ma.shape)

        cat = torch.cat([mx, ma], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        out_put = self.fc_out(x)

        return out_put
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))




class MASAC:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr, 
                 target_entropy, tau, gamma, device, agent_num):
        self.actor = PolicyNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1 = QvalueNet(state_dim*agent_num, hidden_dim*agent_num, action_dim*agent_num).to(device)    
        self.critic_2 = QvalueNet(state_dim*agent_num, hidden_dim*agent_num, action_dim*agent_num).to(device)    
        self.target_critic_1 = QvalueNet(state_dim*agent_num, hidden_dim*agent_num, action_dim*agent_num).to(device)    
        self.target_critic_2 = QvalueNet(state_dim*agent_num, hidden_dim*agent_num, action_dim*agent_num).to(device)    
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
        self.uav_num = agent_num
        self.replay_buffer = ReplayBuffer(capacity=100000)
   
    
    def calc_target(self, rewards, next_states, multi_next_states, dones):
        # print("Shape of rewards:", rewards.shape)
        # print("Shape of next_states:", next_states.shape)
        # print("Shape of dones:", dones.shape)

         # 为每个智能体分别生成动作和 log_prob
        all_next_actions = []
        all_log_probs = []

         # 遍历所有智能体
        for agent_id in range(self.uav_num):
            # 获取每个智能体的局部状态
            # agent_next_state = next_states[:, agent_id * self.state_dim:(agent_id + 1) * self.state_dim]
            agent_next_state = next_states[:, agent_id, :] 

            # 从该智能体的 actor 网络生成动作和 log_prob
            next_action, log_prob = self.actor(agent_next_state)
            all_next_actions.append(next_action)
            all_log_probs.append(log_prob)


        # 将所有智能体的动作拼接成联合动作
        joint_next_actions = torch.cat(all_next_actions, dim=1)
        
        # 将所有 log_prob 也拼接起来，用于熵计算
        log_prob_sum = torch.cat(all_log_probs, dim=1).sum(dim=1, keepdim=True)
        entropy = -log_prob_sum

        q1_value = self.target_critic_1(multi_next_states, joint_next_actions)
        q2_value = self.target_critic_2(multi_next_states, joint_next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy

        td_target = rewards + self.gamma * next_value * (1-dones)

        return td_target
    
    def take_action(self, state):
        state=torch.tensor(np.array(state),dtype=torch.float).to(self.device)
        action=self.actor(state)[0]
        action = action.cpu().detach().numpy().flatten()
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        multi_state = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        multi_next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        # print("multi_state.shape = ", multi_state.shape)
        # print("multi_action.shape = ", actions.shape)

        td_target = self.calc_target(rewards=rewards, 
                                     next_states=multi_next_states, 
                                     multi_next_states=multi_next_states, 
                                     dones=dones)#TODO

        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(multi_state, actions), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(multi_state, actions), td_target.detach()))

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        all_next_actions = []
        all_log_probs = []
        for agent_id in range(self.uav_num):
            # 提取该智能体的局部状态
            agent_state = multi_state[:, agent_id * self.state_dim:(agent_id + 1) * self.state_dim]

            # 使用该智能体的 actor 网络生成动作和 log_prob
            new_action, log_prob = self.actor(agent_state)
            all_next_actions.append(new_action)
            all_log_probs.append(log_prob)
            
        new_actions = torch.cat(all_next_actions, dim=1)
        log_prob_sum = torch.cat(all_log_probs, dim=1).sum(dim=1, keepdim=True)
        entropy = -log_prob_sum

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

    def save_model(self, base_path, scenario):
        self.actor.save_checkpoint(os.path.join(base_path, f"uav_actor_{scenario}.pth"))
        self.critic_1.save_checkpoint(os.path.join(base_path, f"uav_critic_1_{scenario}.pth"))
        self.critic_2.save_checkpoint(os.path.join(base_path, f"uav_critic_2_{scenario}.pth"))
        self.target_critic_1.save_checkpoint(os.path.join(base_path, f"uav_target_critic_1_{scenario}.pth"))
        self.target_critic_2.save_checkpoint(os.path.join(base_path, f"uav_target_critic_2_{scenario}.pth"))

    def load_model(self, base_path, scenario):
        self.actor.load_checkoint(os.path.join(base_path, f"uav_actor_{scenario}.pth"))
        self.critic_1.load_checkpoint(os.path.join(base_path, f"uav_critic_1_{scenario}.pth"))
        self.critic_2.load_checkpoint(os.path.join(base_path, f"uav_critic_2_{scenario}.pth"))
        self.target_critic_1.load_checkpoint(os.path.join(base_path, f"uav_target_critic_1_{scenario}.pth"))
        self.target_critic_2.load_checkpoint(os.path.join(base_path, f"uav_target_critic_2_{scenario}.pth"))
