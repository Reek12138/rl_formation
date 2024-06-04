# -*- coding: utf-8 -*-
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils import agent_selector
from gym.spaces import Box, Discrete
import numpy as np
from pettingzoo.mpe import simple_adversary_v3
from .obstacle import obstacle
from .circle_agent_sac import circle_agent
import matplotlib.pyplot as plt
from math import sin, cos, tan, pi, sqrt, log
from .rvo_inter import rvo_inter



LEADER_MAX_LINEAR_VEL = 2

class CustomEnv:
    metadata = {'render.modes': ['human']}

    def __init__(self, width=100, height=100, num_obstacles=15, agent_radius=5, safe_theta = 2,
                 target_pos = np.array([275, 275]), delta = 0.1, memo_size=100000):
        super().__init__()
        
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.obs_radius = 5
        self.agent_radius = agent_radius
        self.fig = None
        self.ax = None
        
        self.display_time = delta
        self.safe_theta = safe_theta
        self.rvo_inter = rvo_inter()

        # 生成circle_agent实例列表
        self.leader_agent = circle_agent(self, pos=[25, 25], vel=[0,0], orientation=0, memo_size=memo_size,
                                         state_dim=12 + self.num_obstacles * 2,
                                         action_dim=15,
                                         alpha=1e-4,
                                         beta=1e-4,
                                         hidden_dim=600,
                                         gamma=0.99,
                                         tau=0.01,
                                         batch_size=512,
                                         target_entropy= -log(15))

       

        # 生成obstacle实例列表
        self.obstacles = {
            f"obstacle_{i}":obstacle(radius=self.obs_radius,
                                     pos_x=np.random.rand() * self.width *(2/3) + 50,
                                     pos_y=np.random.rand() * self.height *(2/3) + 50,
                                     safe_theta= safe_theta
                                     )for i in range(self.num_obstacles)
        }
        self._check_obs()
        self.leader_target_pos = target_pos#目标位置
        self.reset()

    def _check_obs(self):
        """ 确保障碍物不重复"""
        obstacles_keys = list(self.obstacles.keys())
        obstacles_list = list(self.obstacles.values())
        for i, obs in enumerate(obstacles_list):
            for j in range(i + 1, len(obstacles_list)):
                obs2 = obstacles_list[j]
                dis = np.linalg.norm(obs.position() - obs2.position())
                while dis < 2 * self.obs_radius:
                    key = obstacles_keys[j]
                    self.obstacles[key].pos_x = np.random.rand() * self.width*0.8
                    self.obstacles[key].pos_y = np.random.rand() * self.height*0.8
                    dis = np.linalg.norm(obs.position() - self.obstacles[key].position())


    def reset(self):

        self.leader_target_pos = [10+np.random.rand()*self.width*0.8, 10+np.random.rand()*self.height*0.8]

        self.leader_agent.set_position(10+np.random.rand()*self.width*0.8, 10+np.random.rand()*self.height*0.8)

        while  np.linalg.norm(np.array(self.leader_target_pos) - np.array(self.leader_agent.pos)) < 10:
            self.leader_agent.set_position(10+np.random.rand()*self.width*0.8, 10+np.random.rand()*self.height*0.8)

        target_distance =  np.linalg.norm(np.array(self.leader_target_pos) - np.array(self.leader_agent.pos))

        self.leader_agent.set_vel(0,0)
        self.leader_agent.orientation = np.random.rand()*2*np.pi
        self.leader_agent.done = False
        self.leader_agent.target = False
        
        observations = self.observe_leader(self.leader_agent, self.leader_target_pos, [0,0], target_distance)
        target_info = self.leader_agent.target

        return observations, self.leader_agent.done

    def step(self, action, num_step,target_distance):
        """输入leader_action[线速度，角速度]
                        }"""
        
        self._apply_leader_action(self.leader_agent,action, self.leader_target_pos)
        
        observation = self.observe_leader(self.leader_agent, self.leader_target_pos, action, target_distance)
        reward = self._calculate_leader_reward(agent_id="leader_agent", agent=self.leader_agent, formation_target=self.leader_target_pos, action = action, t=num_step)
        done = self.leader_agent.done
        target = self.leader_agent.target

        return observation, reward, done, target
    
    def _apply_leader_action(self, agent, action, target):#TODO
        """假设输入的动作是[线速度m/s,角速度 弧度]"""
        linear_vel = action[0] 
        angular_vel = action[1] 
        # new_orientation = agent.orientation + action[1]*self.display_time
        new_orientation = agent.orientation + angular_vel*self.display_time
        new_orientation = new_orientation % (2*np.pi)

        dx = linear_vel * self.display_time * cos(new_orientation)
        dy = linear_vel * self.display_time * sin(new_orientation)
        x = agent.pos[0] + dx
        y = agent.pos[1] + dy
        new_pos = [x,y]

        target_dis  = sqrt((x-target[0])**2 + (y-target[1])**2)
        if target_dis < 10:#到达目标点
            agent.target = True

        if x<0 or x>100 or y<0 or y>100:
            flag = True
        else:
            flag = False

        if not self._check_obs_collision(agent, new_pos)and not flag and not agent.target:
        # if not flag:
            agent.set_position(x, y)  
            
            agent.orientation = new_orientation
        else:
            agent.done = True

   
    def _check_obs_collision(self,current_agent,new_pos):
        """检查智能体是否与障碍物碰撞"""
        for obs in self.obstacles.values():
            obs_pos = [obs.pos_x, obs.pos_y]
            if np.linalg.norm(np.array(new_pos) -np.array(obs_pos)) < self.obs_radius:
                return True
        return False
    
   
    def observe_leader(self, agent, target, action, target_distance):
        """领航者自身位置，领航者与目标的距离和角度，与最近障碍物之间的距离（还有啥要加的？TODO）"""
        
        target_dis, target_angle = CustomEnv.calculate_relative_distance_and_angle(agent.pos, target)
        self_pos_ = [agent.pos[0] / self.width, agent.pos[1] / self.height]

        xy_dis = [(agent.pos[0]-0) / self.width, (100-agent.pos[0]) / self.width , (agent.pos[1]-0) / self.height, (100-agent.pos[1]) / self.height]
        # action[0] = (action[0] + 1)/2
        # action[1] = (action[1] + 1)/2
        
        obs_distance_angle = []

        for obs_id, obs in self.obstacles.items():
            obs_pos = [obs.pos_x, obs.pos_y]
            obs_distance, obs_angle = CustomEnv.calculate_relative_distance_and_angle(agent.pos, obs_pos)
            obs_distance_ = obs_distance/(self.width * 1.414)
            obs_angle_ = obs_angle/(2*np.pi)
            obs_distance_angle.extend([obs_distance_, obs_angle_])
        
        # observation = np.array(self_pos + [target_dis, target_angle] + obs_distance_angle)
        observation = np.array(self_pos_ + 
                               [target_dis/target_distance, target_angle/(2*np.pi)] +
                               [target[0]/self.width, target[1]/self.height]+
                                xy_dis + 
                                [action[0], action[1]]+
                               obs_distance_angle)
        # print(observation)

        return observation

    
    def _find_closest_obstacle(self, agent_pos):
        """这个方法计算最近障碍物的位置"""
        closest_obstacle_pos = None
        min_dist = float('inf')
        for obstacle in self.obstacles.values():
            dist = np.linalg.norm(np.array([obstacle.pos_x, obstacle.pos_y]) - np.array(agent_pos))
            if dist < min_dist:
                min_dist = dist
                closest_obstacle = obstacle
        return closest_obstacle
        

    def render(self, mode='human', display_time = 0.1):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(10,10), dpi=100)
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)
            self.ax.set_aspect('equal')

        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)

        # 绘制智能体
        self.ax.plot(self.leader_agent.pos[0], self.leader_agent.pos[1], 'o', color='purple', markersize=self.agent_radius)
        

        # 绘制障碍物
        for obs in self.obstacles.values():
            self.ax.plot(obs.pos_x, obs.pos_y, 'o', color='red', markersize=self.obs_radius)

        # 绘制目标
        self.ax.plot(self.leader_target_pos[0], self.leader_target_pos[1], 'o', color='green', markersize=8)

        plt.pause(self.display_time)  # 暂停以更新图形
        # plt.show()

    def render_close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None

    

    def _calculate_leader_reward(self, agent_id, agent, formation_target, action, t):#TODO
        
        reward1 = self._caculate_leader_vo_reward(agent, formation_target, action )
        reward2 = self._caculate_formation_reward(agent_id, agent, formation_target, action, t, "leader" )
        reward3 = self._caculate_obstacle_reward(agent_id, agent)
        
        # reward = reward1[2] + reward2
        reward =  reward2 +reward3
        return reward

   
    def _caculate_leader_vo_reward(self, agent, target, agent_action):
        vx = agent.orientation * cos(agent.orientation)
        vy = agent.orientation * sin(agent.orientation)
        robot_state = [agent.pos[0], agent.pos[1], vx, vy, self.agent_radius]
        nei_state_list = []
        obs_cir_list = []
        for obs in self.obstacles.values():
            obs_state = [obs.pos_x, obs.pos_y, obs.xy_vel[0], obs.xy_vel[1], self.obs_radius]
            obs_cir_list.append(obs_state)
        obs_line_list = []
        action = [vx, vy]
        vo_flag, min_exp_time, min_dis = self.rvo_inter.config_vo_reward(robot_state=robot_state,
                                                                                                                                        nei_state_list=nei_state_list,
                                                                                                                                        obs_cir_list=obs_cir_list,
                                                                                                                                        obs_line_list=obs_line_list,
                                                                                                                                        action=action)
        reward = [vo_flag, min_exp_time, min_dis]
        return reward
    
    
    def _caculate_formation_reward(self, check_agent_id, check_agent, formation_target, action, t, agent_type):
        """和编队目标之间的距离"""
        dis, angle = CustomEnv.calculate_relative_distance_and_angle(check_agent.pos, formation_target)

        # 设置最小阈值
        min_dis_threshold = 15
        if dis < min_dis_threshold:
            dis = min_dis_threshold

        # if np.isnan(dis) or np.isinf(dis):
        #     print(f"Invalid distance detected! dis: {dis}")
        #     return -100  # 或其他合理的默认值

        # if np.isnan(action).any() or np.isinf(action).any():
        #     print(f"Invalid action detected! action: {action}")
        #     return -100  # 或其他合理的默认值

        if agent_type == "follower":
            reward = - dis / 3  # TODO要不要考虑归一化
        
        if agent_type == "leader":
            if check_agent.done:
                return -100
            
            if check_agent.target:
                return 200
            
            dis_ = dis / 1.414
            reward = (((action[0])/2)*LEADER_MAX_LINEAR_VEL - abs(action[1])*0.5*np.pi +(1- (dis_ / 100))*10 ) / 3

            # if np.isnan(reward) or np.isinf(reward):
            #     print(f"NaN or Inf detected in reward calculation! reward: {reward}, dis: {dis}, action: {action}")
            #     reward = -100  # 或其他合理的默认值

            return reward

    def _caculate_obstacle_reward(self, agent_id, agent):
        reward = 0
        for obs in self.obstacles.values():
            dis, angle = CustomEnv.calculate_relative_distance_and_angle(agent.pos, [obs.pos_x, obs.pos_y])
            dis_ = dis/(100 * 1.414)
            reward += dis_
        reward = 1-reward/self.num_obstacles
        return reward


    @staticmethod
    def calculate_relative_distance_and_angle(pos1, pos2):
        """
        计算两点之间的相对距离和角度

        参数:
        - pos1: 第一个点的位置 (numpy数组或列表 [x, y])
        - pos2: 第二个点的位置 (numpy数组或列表 [x, y])

        返回值:
        - distance: 两点之间的距离
        - angle: 从pos1到pos2的角度（弧度）
        """
        # 计算相对位置向量
        relative_pos = np.array(pos2) - np.array(pos1)
        
        # 计算距离
        distance = np.linalg.norm(relative_pos)
        
        # 计算角度，使用arctan2来得到正确的象限
        angle = np.arctan2(relative_pos[1], relative_pos[0])
        
        return distance, angle