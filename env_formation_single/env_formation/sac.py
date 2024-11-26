import torch
from torch import nn 
from torch.nn import functional as F
import numpy as np
import random
import math
import os
from torch.distributions import Normal
import collections
from torch.utils.tensorboard import SummaryWriter


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity  # 树的最大容量
        self.tree = np.zeros(2 * capacity - 1)  # 用于存储优先级的树
        self.data = np.zeros(capacity, dtype=object)  # 用于存储经验的数组
        self.write_index = 0  # 当前写入的位置
        self.size = 0  # 当前存储的经验数量

    def add(self, priority, data):
        tree_index = self.write_index + self.capacity - 1
        self.data[self.write_index] = data
        self.update(tree_index, priority)
        self.write_index += 1
        if self.write_index >= self.capacity:
            self.write_index = 0  # 循环覆盖
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:  # 更新父节点
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, value):
        parent_index = 0
        while True:
            left_child = 2 * parent_index + 1
            right_child = left_child + 1
            if left_child >= len(self.tree):
                leaf_index = parent_index
                break
            if value <= self.tree[left_child]:
                parent_index = left_child
            else:
                value -= self.tree[left_child]
                parent_index = right_child
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    def total_priority(self):
        return self.tree[0]  # 根节点的值是总优先级
    

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

# class ReplayBuffer:
#     def __init__(self, capacity, alpha=0.6):
#         self.tree = SumTree(capacity)
#         self.alpha = alpha  # 控制优先级的偏重程度
#         self.epsilon = 1e-5  # 防止优先级为 0
#         self.max_priority = 1.0  # 初始化的最大优先级

#     def add(self, state, action, reward, next_state, done):
#         experience = (state, action, reward, next_state, done)
#         priority = self.max_priority  # 新经验使用当前最大优先级
#         self.tree.add(priority, experience)

#     def sample(self, batch_size, beta=0.4):
#         batch = []
#         idxs = []
#         priorities = []
#         segment = self.tree.total_priority() / batch_size

#         for i in range(batch_size):
#             value = random.uniform(i * segment, (i + 1) * segment)
#             idx, priority, data = self.tree.get_leaf(value)
#             batch.append(data)
#             idxs.append(idx)
#             priorities.append(priority)

#         # 计算重要性采样权重
#         sampling_probabilities = np.array(priorities) / self.tree.total_priority()
#         is_weights = np.power(len(self.tree.data) * sampling_probabilities, -beta)
#         is_weights /= is_weights.max()  # 归一化

#         states, actions, rewards, next_states, dones = zip(*batch)
#         return (
#             np.array(states),
#             np.array(actions),
#             np.array(rewards),
#             np.array(next_states),
#             np.array(dones),
#             idxs,
#             is_weights,
#         )

#     def update_priorities(self, idxs, priorities):
#         for idx, priority in zip(idxs, priorities):
#             self.tree.update(idx, priority + self.epsilon)
#         self.max_priority = max(self.max_priority, max(priorities))
    
#     def clear(self):
#         self.tree = SumTree(self.tree.capacity)  # 重置 SumTree
#         self.max_priority = 1.0 


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)

         # 使用 Xavier 初始化权重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_std.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        # std = F.softplus(self.fc_std(x)) + 1e-6
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)

        # log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        log_prob = log_prob - torch.log(torch.clamp(1 - torch.tanh(action).pow(2), min=1e-6))
        log_prob = log_prob.sum(dim = -1, keepdim=True)

        return action, log_prob
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpont_file):
        self.load_state_dict(torch.load(checkpont_file))



class QvalueNet(nn.Module):
    def __init__(self, multi_state_dim, multi_hidden_dim, multi_action_dim):
        super(QvalueNet, self).__init__()
        self.fc1 = nn.Linear(multi_state_dim + multi_action_dim, multi_hidden_dim)
        self.fc2 = nn.Linear(multi_hidden_dim, multi_hidden_dim)
        self.fc_out = nn.Linear(multi_hidden_dim, 1)

        # 使用 He 初始化权重
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc_out.weight)  # 输出层通常不需要特定激活函数的考虑
    
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




class SAC:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr, 
                 target_entropy, tau, gamma, device, agent_num):
        self.actor = PolicyNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1 = QvalueNet(state_dim, hidden_dim, action_dim).to(device)    
        self.critic_2 = QvalueNet(state_dim, hidden_dim, action_dim).to(device)    
        self.target_critic_1 = QvalueNet(state_dim, hidden_dim, action_dim).to(device)    
        self.target_critic_2 = QvalueNet(state_dim, hidden_dim, action_dim).to(device)    
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_1_optimizer = torch.optim.AdamW(self.critic_1.parameters(), lr=critic_lr, weight_decay=1e-5)

        # self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.AdamW(self.critic_2.parameters(), lr=critic_lr, weight_decay=1e-5)
        
        # self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        # self.log_alpha.requires_grad=True
        self.log_alpha = torch.tensor(np.log(0.1), dtype=torch.float, requires_grad=True, device=device)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy  =target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.state_dim = state_dim
        self.uav_num = agent_num
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.training_step = 0
        self.actor_update_interval = 2

        # 在类的初始化函数中初始化 TensorBoard
        self.writer = SummaryWriter(log_dir="./sac_logs")
        # 初始化损失记录字典
        self.losses = {
            'follower_critic_1_loss': [],
            'follower_critic_2_loss': [],
            'follower_actor_loss': [],
            'follower_alpha_loss': []
        }
   
    
    def calc_target(self,rewards,next_states,dones):  #计算目标Q值
        next_actions,log_prob=self.actor(next_states)
        #计算熵，注意这里是有个负号的
        entropy=-log_prob
        q1_value=self.target_critic_1(next_states,next_actions)
        q2_value=self.target_critic_2(next_states,next_actions)
        #注意entropy自带负号
        next_value=torch.min(q1_value,q2_value) + self.log_alpha.exp() * entropy
         # 打印调试信息
        # print(f"q1_value shape: {q1_value.shape}, q2_value shape: {q2_value.shape}")
    
        # print(f"min_next_q_values shape: {next_value.shape}")
    
        td_target=rewards + self.gamma * next_value *(1-dones)
        # 打印调试信息
        # print(f"td_target shape: {td_target.shape}")
        return td_target
    
    def take_action(self, state):
        state=torch.tensor(np.array(state),dtype=torch.float).to(self.device)
        action=self.actor(state)[0]
        action = action.cpu().detach().numpy().flatten()
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    # def update(self,transition_dict):
    #     states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
    #     # actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
    #     actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
    #     rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
    #     next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
    #     dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        
    #     # 确保拼接之前的维度匹配
    #     # print(f"states shape: {states.shape}, actions shape: {actions.shape}")
    
    #     #更新两个Q网络
    #     td_target=self.calc_target(rewards,next_states,dones)
    #     #Q网络输出值和目标值的均方差
    #     critic_1_loss=torch.mean(F.mse_loss(self.critic_1(states,actions),td_target.detach()))
    #     critic_2_loss=torch.mean(F.mse_loss(self.critic_2(states,actions),td_target.detach()))
    #     self.critic_1_optimizer.zero_grad()
    #     critic_1_loss.backward()
    #     self.critic_1_optimizer.step()
    #     self.critic_2_optimizer.zero_grad()
    #     critic_2_loss.backward()
    #     self.critic_2_optimizer.step()
        
    #     #更新策略网络
    #     new_actions, log_prob=self.actor(states)
    #     entropy= -log_prob
    #     q1_value=self.critic_1(states,new_actions)
    #     q2_value=self.critic_2(states,new_actions)
    #     #最大化价值，所以误差为价值函数加负号
    #     actor_loss=torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value,q2_value))
    #     self.actor_optimizer.zero_grad()
    #     actor_loss.backward()
    #     self.actor_optimizer.step()
        
    #     #更新alpha值
    #     #利用梯度下降自动调整熵正则项
    #     alpha_loss=torch.mean((entropy - self.target_entropy).detach() *self.log_alpha.exp())
    #     self.log_alpha_optimizer.zero_grad()
    #     alpha_loss.backward()
    #     self.log_alpha_optimizer.step()
        
    #     #软更新目标网络
    #     self.soft_update(self.critic_1,self.target_critic_1)
    #     self.soft_update(self.critic_2,self.target_critic_2)
    #     # 在 TensorBoard 中记录损失
    #     step = len(self.losses['follower_critic_1_loss'])  # 当前步数
    #     self.writer.add_scalar('Loss/follower_Critic1', critic_1_loss.item(), step)
    #     self.writer.add_scalar('Loss/follower_Critic2', critic_2_loss.item(), step)
    #     self.writer.add_scalar('Loss/follower_Actor', actor_loss.item(), step)
    #     self.writer.add_scalar('Loss/follower_Alpha', alpha_loss.item(), step)
    def update(self, transition_dict):
        # 数据转换到张量
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # Critic 更新
        self.update_critics(states, actions, rewards, next_states, dones)

        # Actor 和 Alpha 每隔一定步数更新一次
        if self.training_step % self.actor_update_interval == 0:
            self.update_actor(states)
            # self.update_alpha(states)
            new_actions, log_prob = self.actor(states)  # 重新计算 log_prob
            self.update_alpha(-log_prob)

        # 软更新目标网络
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        # 记录步数
        self.training_step += 1

    def update_critics(self, states, actions, rewards, next_states, dones):
        """更新 Critic 网络"""
        td_target = self.calc_target(rewards, next_states, dones)
        # critic_1_loss = F.mse_loss(self.critic_1(states, actions), td_target.detach())
        critic_1_loss = F.mse_loss(self.critic_1(states, actions), td_target.detach())
        # critic_2_loss = F.mse_loss(self.critic_2(states, actions), td_target.detach())
        critic_2_loss = F.mse_loss(self.critic_2(states, actions), td_target.detach())

        # 优化 Critic 网络
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 在 TensorBoard 中记录 Critic 损失
        step = self.training_step
        self.writer.add_scalar('Loss/Critic1', critic_1_loss.item(), step)
        self.writer.add_scalar('Loss/Critic2', critic_2_loss.item(), step)

    def update_actor(self, states):
        """更新 Actor 网络"""
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob

        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)

        # 计算 Actor 损失
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))

        # 优化 Actor 网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 在 TensorBoard 中记录 Actor 损失
        # step = self.training_step
        step = self.training_step // self.actor_update_interval
        self.writer.add_scalar('Loss/Actor', actor_loss.item(), step)

    def update_alpha(self, entropy):
        """更新 Alpha 值"""
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())

        # 优化 Alpha
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # 在 TensorBoard 中记录 Alpha 损失
        # step = self.training_step
        step = self.training_step // self.actor_update_interval
        self.writer.add_scalar('Loss/Alpha', alpha_loss.item(), step)

        

    def save_model(self, base_path, scenario):
        self.actor.save_checkpoint(os.path.join(base_path, f"uav_actor_{scenario}.pth"))
        self.critic_1.save_checkpoint(os.path.join(base_path, f"uav_critic_1_{scenario}.pth"))
        self.critic_2.save_checkpoint(os.path.join(base_path, f"uav_critic_2_{scenario}.pth"))
        self.target_critic_1.save_checkpoint(os.path.join(base_path, f"uav_target_critic_1_{scenario}.pth"))
        self.target_critic_2.save_checkpoint(os.path.join(base_path, f"uav_target_critic_2_{scenario}.pth"))
        torch.save(self.log_alpha, os.path.join(base_path, f"log_alpha_{scenario}.pth"))

    def load_model(self, base_path, scenario):
        # print("Loading model from:", os.path.join(base_path, f"uav_target_critic_1_{scenario}.pth"))

        self.actor.load_checkpoint(os.path.join(base_path, f"uav_actor_{scenario}.pth"))
        self.critic_1.load_checkpoint(os.path.join(base_path, f"uav_critic_1_{scenario}.pth"))
        self.critic_2.load_checkpoint(os.path.join(base_path, f"uav_critic_2_{scenario}.pth"))
        self.target_critic_1.load_checkpoint(os.path.join(base_path, f"uav_target_critic_1_{scenario}.pth"))
        self.target_critic_2.load_checkpoint(os.path.join(base_path, f"uav_target_critic_2_{scenario}.pth"))
        self.log_alpha = torch.load(os.path.join(base_path, f"log_alpha_{scenario}.pth")).to(self.device)
        self.log_alpha.requires_grad = True  # 确保重新加载后继续优化
