# 处理离散问题的模型
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import collections
import random
import math
import os
from torch.distributions import Normal
 
# ----------------------------------------- #
# 经验回放池
# ----------------------------------------- #
 
class ReplayBuffer:
    def __init__(self, capacity):  # 经验池容量
        self.buffer = collections.deque(maxlen=capacity)  # 队列，先进先出
    # 经验池增加
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    # 随机采样batch组
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # 取出这batch组数据
        state, action, reward, next_state, done = zip(*transitions)
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done)
        )
    # 当前时刻的经验池容量
    def size(self):
        return len(self.buffer)
    
    def clear(self):
        """清空经验池"""
        self.buffer.clear()
 
# # ----------------------------------------- #
# # 策略网络
# # ----------------------------------------- #
 
# class PolicyNet(nn.Module):
#     def __init__(self, n_states, n_hiddens, n_actions):
#         super(PolicyNet, self).__init__()
#         self.fc1 = nn.Linear(n_states, n_hiddens)
#         self.fc2 = nn.Linear(n_hiddens, n_actions)
#     # 前向传播
#     def forward(self, x):  # 获取当前状态下的动作选择概率
#         x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
#         x = F.relu(x)
#         x = self.fc2(x)  # [b,n_hiddens]-->[b,n_actions]
#         # 每个状态下对应的每个动作的动作概率
#         x = F.softmax(x, dim=1)  # [b,n_actions]
#         return x
    
#     def save_checkpoint(self, checkpoint_file):
#         torch.save(self.state_dict(), checkpoint_file)

#     def load_checkpoint(self, checkpoint_file):
#         self.load_state_dict(torch.load(checkpoint_file))
 
# # ----------------------------------------- #
# # 价值网络
# # ----------------------------------------- #
 
# class ValueNet(nn.Module):
#     def __init__(self, n_states, n_hiddens, n_actions):
#         super(ValueNet, self).__init__()
#         self.fc1 = nn.Linear(n_states, n_hiddens)
#         self.fc2 = nn.Linear(n_hiddens, n_actions)
#     # 当前时刻的state_value
#     def forward(self, x):  
#         x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
#         x = F.relu(x)  
#         x = self.fc2(x)  # [b,n_hiddens]-->[b,n_actions]
#         return x
    
#     def save_checkpoint(self, checkpoint_file):
#         torch.save(self.state_dict(), checkpoint_file)

#     def load_checkpoint(self, checkpoint_file):
#         self.load_state_dict(torch.load(checkpoint_file))
 

class PolicyNetContinuous(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(PolicyNetContinuous,self).__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc_mu=nn.Linear(hidden_dim,action_dim)
        self.fc_std=nn.Linear(hidden_dim,action_dim)
        
        
    def forward(self,x):
        x=F.relu(self.fc1(x))
        mu=self.fc_mu(x)
        std=F.softplus(self.fc_std(x))
        #创建一个正态分布对象，作为动作空间
        dist=Normal(mu,std)
        #动作空间采样，得到样本，样本是动作值，代表连续空间中对应的动作
        normal_sample=dist.rsample()  #rsample()是重参数化采样
        #计算样本的对数概率密度
        log_prob=dist.log_prob(normal_sample)
        #将动作约数在[-1,1]
        action=torch.tanh(normal_sample)
        #计算tanh_normal分布的对数概率密度
#限制动作范围会影响到动作的概率密度函数。这是因为 tanh 函数的导数在边界点上接近于零，这可能导致在这些点上计算的概率密度非常小，甚至接近于零。这会导致梯度消失，从而影响模型的训练效果。
#为了解决这个问题，可以使用公式 log(1 - tanh^2(x) + ε) 来重新计算对数概率密度，其中 ε 是一个较小的常数（在这里是 1e-7），用于避免取对数时的除零错误。这样可以保持对数概率密度的合理值，并避免梯度消失的问题。因此，在该代码中，使用该公式重新计算 log_prob。
        log_prob=log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)


        # # 重新计算 log_prob 以考虑 tanh 的影响
        # log_prob = log_prob - torch.log(1 - torch.tanh(normal_sample).pow(2) + 1e-7)
        # log_prob = log_prob.sum(dim=-1, keepdim=True)  # 累加各个动作维度的 log_prob
        
        #得到动作的范围
        # action=action * self.action_bound
        return action, log_prob
 
 
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))
 
 
class QValueNetContinuous(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(QValueNetContinuous,self).__init__()
        self.fc1=nn.Linear(state_dim+action_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc_out=nn.Linear(hidden_dim,1)
        
    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        output = self.fc_out(x)
        
        # # 打印输入和输出的形状
        # print(f"Input state shape: {x.shape}, action shape: {a.shape}")
        # print(f"Concatenated input shape: {cat.shape}")
        # print(f"Output shape: {output.shape}")
        
        return output
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))
# ----------------------------------------- #
# 模型构建
# ----------------------------------------- #

class SACContinuous:
    """处理连续动作的SAC算法"""
    def __init__(self,state_dim,hidden_dim,action_dim,actor_lr,critic_lr,alpha_lr,target_entropy,tau,gamma,device):
        self.actor=PolicyNetContinuous(state_dim,hidden_dim,action_dim).to(device)  #策略网络
        self.critic_1=QValueNetContinuous(state_dim,hidden_dim,action_dim).to(device)  #第一个Q网络
        self.critic_2=QValueNetContinuous(state_dim,hidden_dim,action_dim).to(device)  #第二个Q网络
        self.target_critic_1=QValueNetContinuous(state_dim,hidden_dim,action_dim).to(device)  #第一个目标Q网络
        self.target_critic_2=QValueNetContinuous(state_dim,hidden_dim,action_dim).to(device)  #第二个目标Q网络
        #令目标Q网络的参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_1_optimizer=torch.optim.Adam(self.critic_1.parameters(),lr=critic_lr)
        self.critic_2_optimizer=torch.optim.Adam(self.critic_2.parameters(),lr=critic_lr)
        #使用alpha的log值，可以使训练结果比较稳定
        self.log_alpha=torch.tensor(np.log(0.01),dtype=torch.float)
        self.log_alpha.requires_grad=True  #可以对alpha求梯度
        self.log_alpha_optimizer=torch.optim.Adam([self.log_alpha],lr=alpha_lr)
        self.target_entropy=target_entropy  #目标熵的大小
        self.gamma=gamma
        self.tau=tau
        self.device=device
        
    def take_action(self,state):
        state=torch.tensor(np.array(state),dtype=torch.float).to(self.device)
        action=self.actor(state)[0]
        action = action.cpu().detach().numpy().flatten()
        return action
    
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
    
    def soft_update(self,net,target_net):
        for param_target, param in zip(target_net.parameters(),net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
            
    def update(self,transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        # actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        
        # 确保拼接之前的维度匹配
        # print(f"states shape: {states.shape}, actions shape: {actions.shape}")
    
        #更新两个Q网络
        td_target=self.calc_target(rewards,next_states,dones)
        #Q网络输出值和目标值的均方差
        critic_1_loss=torch.mean(F.mse_loss(self.critic_1(states,actions),td_target.detach()))
        critic_2_loss=torch.mean(F.mse_loss(self.critic_2(states,actions),td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        
        #更新策略网络
        new_actions, log_prob=self.actor(states)
        entropy= -log_prob
        q1_value=self.critic_1(states,new_actions)
        q2_value=self.critic_2(states,new_actions)
        #最大化价值，所以误差为价值函数加负号
        actor_loss=torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value,q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        #更新alpha值
        #利用梯度下降自动调整熵正则项
        alpha_loss=torch.mean((entropy - self.target_entropy).detach() *self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        
        #软更新目标网络
        self.soft_update(self.critic_1,self.target_critic_1)
        self.soft_update(self.critic_2,self.target_critic_2)

    def save_model(self, base_path, scenario):
        self.actor.save_checkpoint(os.path.join(base_path, f"leader_agent_actor_{scenario}.pth"))
        self.critic_1.save_checkpoint(os.path.join(base_path, f"leader_agent_critic1_{scenario}.pth"))
        self.critic_2.save_checkpoint(os.path.join(base_path, f"leader_agent_critic2_{scenario}.pth"))
        self.target_critic_1.save_checkpoint(os.path.join(base_path, f"leader_agent_target_critic1_{scenario}.pth"))
        self.target_critic_2.save_checkpoint(os.path.join(base_path, f"leader_agent_target_critic2_{scenario}.pth"))

    def load_model(self, base_path, scenario):
        self.actor.load_checkpoint(os.path.join(base_path, f"leader_agent_actor_{scenario}.pth"))
        self.critic_1.load_checkpoint(os.path.join(base_path, f"leader_agent_critic1_{scenario}.pth"))
        self.critic_2.load_checkpoint(os.path.join(base_path, f"leader_agent_critic2_{scenario}.pth"))
        self.target_critic_1.load_checkpoint(os.path.join(base_path, f"leader_agent_target_critic1_{scenario}.pth"))
        self.target_critic_2.load_checkpoint(os.path.join(base_path, f"leader_agent_target_critic2_{scenario}.pth"))


class circle_agent():
    def __init__(self, radius = 5, pos=[25, 25], vel=[0, 0], orientation=np.pi/4,
                 memo_size = 100000, state_dim = 40, action_dim = 2, alpha = 1e-4, beta = 1e-4,alpha_lr = 1e-4,
                 hidden_dim = 600, gamma = 0.99, tau = 0.01, batch_size = 512, target_entropy = -math.log(2)):
        self.radius = radius
        self.pos = pos
        self.vel = vel
        self.orientation = orientation
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

        # self.action = [[0, -np.pi/2], [1, -np.pi/2], [2, -np.pi/2],
        #                [0, -np.pi/4], [1, -np.pi/4], [2, -np.pi/4],
        #                [0, 0], [1, 0], [2, 0],
        #                [0, np.pi/4], [1, np.pi/4], [2, np.pi/4],
        #                [0, np.pi/2], [1, np.pi/2], [2, np.pi/2]]

        self.sac_network = SACContinuous(state_dim=self.state_dim, hidden_dim=self.hidden_dim, action_dim=self.action_dim,
                                        actor_lr=self.alpha,critic_lr=self.beta,alpha_lr=self.alpha_lr,
                                         target_entropy=self.target_entropy,tau=self.tau,gamma=self.gamma,
                                          device=self.device )
        self.replay_buffer = ReplayBuffer(capacity=self.memo_size)
        

    def set_position(self, x, y):
        self.pos[0] = x
        self.pos[1] = y
    
    def set_vel(self, v):
        self.vel = v

    def position(self):
        return np.array([self.pos])

    