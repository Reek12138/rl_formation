# 处理离散问题的模型
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import collections
import random
import math
 
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
        return np.array(state), action, reward, np.array(next_state), done
    # 当前时刻的经验池容量
    def size(self):
        return len(self.buffer)
 
# ----------------------------------------- #
# 策略网络
# ----------------------------------------- #
 
class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_actions)
    # 前向传播
    def forward(self, x):  # 获取当前状态下的动作选择概率
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_actions]
        # 每个状态下对应的每个动作的动作概率
        x = F.softmax(x, dim=1)  # [b,n_actions]
        return x
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))
 
# ----------------------------------------- #
# 价值网络
# ----------------------------------------- #
 
class ValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_actions)
    # 当前时刻的state_value
    def forward(self, x):  
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)  
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_actions]
        return x
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))
 
# ----------------------------------------- #
# 模型构建
# ----------------------------------------- #
 
class SAC:
    def __init__(self, n_states, n_hiddens, n_actions,
                 actor_lr, critic_lr, alpha_lr,
                 target_entropy, tau, gamma, device, action):
        
        # 实例化策略网络
        self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)
        # 实例化第一个价值网络--预测
        self.critic_1 = ValueNet(n_states, n_hiddens, n_actions).to(device)
        # 实例化第二个价值网络--预测
        self.critic_2 = ValueNet(n_states, n_hiddens, n_actions).to(device)
        # 实例化价值网络1--目标
        self.target_critic_1 = ValueNet(n_states, n_hiddens, n_actions).to(device)
        # 实例化价值网络2--目标
        self.target_critic_2 = ValueNet(n_states, n_hiddens, n_actions).to(device)
 
        # 预测和目标的价值网络的参数初始化一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 目标网络的优化器
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
 
        # 初始化可训练参数alpha
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        # alpha可以训练求梯度
        self.log_alpha.requires_grad = True
        # 定义alpha的优化器
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
 
        # 属性分配
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.action = action
    
    # 动作选择
    def take_action(self, state):  # 输入当前状态 [n_states]
        # 维度变换 numpy[n_states]-->tensor[1,n_states]
        state = torch.tensor(state[np.newaxis,:], dtype=torch.float).to(self.device)
        # 预测当前状态下每个动作的概率  [1,n_actions]
        probs = self.actor(state)
        # 构造与输出动作概率相同的概率分布
        action_dist = torch.distributions.Categorical(probs)
        # 从当前概率分布中随机采样tensor-->int
        action_num = action_dist.sample().item()
        action = self.action[action_num]
        return action
    
    # 计算目标，当前状态下的state_value
    def calc_target(self, rewards, next_states, dones):
        # 策略网络预测下一时刻的state_value  [b,n_states]-->[b,n_actions]
        next_probs = self.actor(next_states)
        # 对每个动作的概率计算ln  [b,n_actions]
        next_log_probs = torch.log(next_probs + 1e-8)
        # 计算熵 [b,1]
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdims=True)
        # 目标价值网络，下一时刻的state_value [b,n_actions]
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        # 取出最小的q值  [b, 1]
        min_qvalue = torch.sum(next_probs * torch.min(q1_value,q2_value), dim=1, keepdims=True)
        # 下个时刻的state_value  [b, 1]
        next_value = min_qvalue + self.log_alpha.exp() * entropy
 
        # 时序差分，目标网络输出当前时刻的state_value  [b, n_actions]
        td_target = rewards + self.gamma * next_value * (1-dones)
        return td_target
    
    # 软更新，每次训练更新部分参数
    def soft_update(self, net, target_net):
        # 遍历预测网络和目标网络的参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # 预测网络的参数赋给目标网络
            param_target.data.copy_(param_target.data*(1-self.tau) + param.data*self.tau)
    
    def save_model(self, filename, scenario):
        self.actor.save_checkpoint(f"{filename}" + f"leader_agent_actor_{scenario}.pth")
        self.critic_1.save_checkpoint(f"{filename}" + f"leader_agent_critic1_{scenario}.pth")
        self.critic_2.save_checkpoint(f"{filename}" + f"leader_agent_critic2_{scenario}.pth")
        self.target_critic_1.save_checkpoint(f"{filename}" + f"leader_agent_target_critic1_{scenario}.pth")
        self.target_critic_2.save_checkpoint(f"{filename}" + f"leader_agent_target_critic2_{scenario}.pth")
    
    def load_model(self, filename, scenario):
        self.actor.load_checkpoint(f"{filename}" + f"leader_agent_actor_{scenario}.pth")
        self.critic_1.load_checkpoint(f"{filename}" + f"leader_agent_critic1_{scenario}.pth")
        self.critic_2.load_checkpoint(f"{filename}" + f"leader_agent_critic2_{scenario}.pth")
        self.target_critic_1.load_checkpoint(f"{filename}" + f"leader_agent_target_critic1_{scenario}.pth")
        self.target_critic_2.load_checkpoint(f"{filename}" + f"leader_agent_target_critic2_{scenario}.pth")
    

    # 模型训练
    def update(self, transition_dict):
        # 提取数据集
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)  # [b,n_states]
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)  # [b,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)  # [b,n_states]
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
 
        # --------------------------------- #
        # 更新2个价值网络
        # --------------------------------- #
 
        # 目标网络的state_value [b, 1]
        td_target = self.calc_target(rewards, next_states, dones)
        # 价值网络1--预测，当前状态下的动作价值  [b, 1]
        critic_1_qvalues = self.critic_1(states).gather(1, actions)
        # 均方差损失 预测-目标
        critic_1_loss = torch.mean(F.mse_loss(critic_1_qvalues, td_target.detach()))
        # 价值网络2--预测
        critic_2_qvalues = self.critic_2(states).gather(1, actions)
        # 均方差损失
        critic_2_loss = torch.mean(F.mse_loss(critic_2_qvalues, td_target.detach()))
        
        # 梯度清0
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        # 梯度反传
        critic_1_loss.backward()
        critic_2_loss.backward()
        # 梯度更新
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
 
        # --------------------------------- #
        # 更新策略网络
        # --------------------------------- #
 
        probs = self.actor(states)  # 预测当前时刻的state_value  [b,n_actions]
        log_probs = torch.log(probs + 1e-8)  # 小于0  [b,n_actions]
        # 计算策略网络的熵>0   [b,1]
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
 
        # 价值网络预测当前时刻的state_value  
        q1_value = self.critic_1(states)  # [b,n_actions]
        q2_value = self.critic_2(states)
        # 取出价值网络输出的最小的state_value  [b,1]
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
 
        # 策略网络的损失
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        # 梯度更新
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
 
        # --------------------------------- #
        # 更新可训练遍历alpha
        # --------------------------------- #
 
        alpha_loss = torch.mean((entropy-self.target_entropy).detach() * self.log_alpha.exp())
        # 梯度更新
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
 
        # 软更新目标价值网络
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

class circle_agent():
    def __init__(self, radius = 5, pos=[25, 25], vel=[0, 0], orientation=np.pi/4,
                 memo_size = 100000, state_dim = 40, action_dim = 15, alpha = 1e-4, beta = 1e-4,alpha_lr = 1e-4,
                 hidden_dim = 600, gamma = 0.99, tau = 0.01, batch_size = 512, target_entropy = -math.log(15)):
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

        self.action = [[0, -np.pi/3], [1, -np.pi/3], [2, -np.pi/3],
                       [0, -np.pi/6], [1, -np.pi/6], [2, -np.pi/6],
                       [0, 0], [1, 0], [2, 0],
                       [0, np.pi/6], [1, np.pi/6], [2, np.pi/6],
                       [0, np.pi/3], [1, np.pi/3], [2, np.pi/3]]

        self.sac_network = SAC(n_states=self.state_dim, n_hiddens=self.hidden_dim, n_actions=self.action_dim,
                               actor_lr=self.alpha, critic_lr=self.beta, alpha_lr=self.alpha_lr,action=self.action,
                               gamma=self.gamma,target_entropy=self.target_entropy,
                               tau=self.tau,device=self.device)
        self.replay_buffer = ReplayBuffer(capacity=self.memo_size)
        

    def set_position(self, x, y):
        self.pos[0] = x
        self.pos[1] = y
    
    def set_vel(self, v1, v2):
        self.vel[0] = v1
        self.vel[1] = v2

    