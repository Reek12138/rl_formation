# -*- coding: utf-8 -*-
# from pettingzoo import AECEnv, ParallelEnv
# from pettingzoo.utils import agent_selector
# from gym.spaces import Box, Discrete
import numpy as np
# from pettingzoo.mpe import simple_adversary_v3
from .obstacle import obstacle
from .circle_agent_sac import circle_agent
import matplotlib.pyplot as plt
from math import sin, cos, tan, pi, sqrt, log
from .rvo_inter import rvo_inter
import matplotlib.patches as patches



LEADER_MAX_LINEAR_VEL = 2

class CustomEnv:
    metadata = {'render.modes': ['human']}

    def __init__(self, width=100, height=100, num_obstacles=12, agent_radius=1, obs_radius = 2,safe_theta = 2,target_radius = 4,
                 target_pos = np.array([50, 50]), delta = 0.1, memo_size=100000):
        super().__init__()
        
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.obs_radius = obs_radius
        self.target_radius = target_radius
        self.agent_radius = agent_radius
        self.fig = None
        self.ax = None
        
        self.display_time = delta
        self.safe_theta = safe_theta
        self.rvo_inter = rvo_inter()

        # 生成circle_agent实例列表
        self.leader_agent = circle_agent(self, pos=[25, 25], vel=[0,0], orientation=0, memo_size=memo_size,
                                         state_dim=7 + self.num_obstacles * 2,
                                         action_dim=2,
                                         alpha=1e-4,
                                         beta=1e-4,
                                         hidden_dim=512,
                                         gamma=0.99,
                                         tau=0.01,
                                         batch_size=512,
                                         target_entropy= -log(2))

       
        self.fix_position =  [
            (25, 25),
            (50, 50),
            (75, 75),
            (75, 25),
            (25, 75),
            (25, 50),
            (50, 25),
            (50, 75),
            (75, 50)
        ]
        # 生成obstacle实例列表
        # self.obstacles = {
        #     f"obstacle_{i}":obstacle(radius=self.obs_radius,
        #                              pos_x=np.random.rand() * self.width *(0.7) + self.width *0.15,
        #                              pos_y=np.random.rand() * self.height *(0.7) + self.height*0.15,
        #                              safe_theta= safe_theta
        #                              )for i in range(self.num_obstacles)
        # }
        self.obstacles = {}
        for i in range(len(self.fix_position)):
            pos_x, pos_y = self.fix_position[i]
            self.obstacles[f"obstacle_{i}"] = obstacle(
                radius=self.obs_radius,
                pos_x=pos_x,
                pos_y=pos_y,
                safe_theta=self.safe_theta
            )

        # 随机位置的障碍物
        for i in range(len(self.fix_position), self.num_obstacles):
            self.obstacles[f"obstacle_{i}"] = obstacle(
                radius=self.obs_radius,
                pos_x=np.random.rand() * self.width * 0.7 + self.width * 0.15,
                pos_y=np.random.rand() * self.height * 0.7 + self.height * 0.15,
                safe_theta=self.safe_theta
            )

        self.leader_target_pos = target_pos#目标位置
        self._check_obs()
        self.reset()

    
    def _check_obs(self):
        """ 确保障碍物不重复 """
        obstacles_keys = list(self.obstacles.keys())
        obstacles_list = list(self.obstacles.values())

        # 仅检查随机位置的障碍物
        random_obstacles = obstacles_list[9:]  # 假设前5个障碍物是固定的

        for i, obs in enumerate(random_obstacles):
            for j in range(i + 1, len(random_obstacles)):
                obs2 = random_obstacles[j]
                dis = np.linalg.norm(obs.position() - obs2.position())
                dis2 = np.linalg.norm(np.array(self.leader_target_pos) - obs2.position())
                while dis < 2 * self.obs_radius + self.agent_radius + self.safe_theta or dis2 < self.obs_radius + self.target_radius + self.agent_radius + self.safe_theta:
                    key = obstacles_keys[9 + j]  # 索引偏移，确保获取随机障碍物的键
                    self.obstacles[key].pos_x = np.random.rand() * self.width * 0.7 + self.width * 0.15
                    self.obstacles[key].pos_y = np.random.rand() * self.height * 0.7 + self.height * 0.15
                    dis = np.linalg.norm(obs.position() - self.obstacles[key].position())
                    dis2 = np.linalg.norm(np.array(self.leader_target_pos) - self.obstacles[key].position())

    def _check_obs_agent(self, agent):
        for obs in self.obstacles.values():
            dis = np.linalg.norm(obs.position() - agent.position())
            if dis <= self.obs_radius + self.agent_radius*2:
                return True
        return False
    
    def _check_obs_target(self, target_pos):
        for obs in self.obstacles.values():
            dis = np.linalg.norm(obs.position() - np.array(target_pos))
            if dis < self.obs_radius + self.target_radius:
                return True
        return False

    def reset(self):


        # 随机位置的障碍物
        for i in range(len(self.fix_position), self.num_obstacles):
            self.obstacles[f"obstacle_{i}"].pos_x = np.random.rand() * self.width * 0.7 + self.width * 0.15
            self.obstacles[f"obstacle_{i}"].pos_y=np.random.rand() * self.height * 0.7 + self.height * 0.15
            
        
        self._check_obs()

        self.leader_target_pos = [self.width*0.1+np.random.rand()*self.width*0.8, self.height*0.1+np.random.rand()*self.height*0.8]

        self.leader_agent.set_position(self.width*0.10+np.random.rand()*self.width*0.8, self.height*0.10+np.random.rand()*self.height*0.8)

        flag2 = self._check_obs_target(self.leader_target_pos)
        while flag2:
            self.leader_target_pos = [self.width*0.1+np.random.rand()*self.width*0.8, self.height*0.1+np.random.rand()*self.height*0.8]
            flag2 = self._check_obs_target(self.leader_target_pos)
        
        flag1 = self._check_obs_agent(self.leader_agent)
        while  np.linalg.norm(np.array(self.leader_target_pos) - np.array(self.leader_agent.pos)) < self.agent_radius + self.target_radius or flag1:
            self.leader_agent.set_position(self.width*0.10+np.random.rand()*self.width*0.8, self.height*0.10+np.random.rand()*self.height*0.8)
            flag1 = self._check_obs_agent(self.leader_agent)

        target_distance =  np.linalg.norm(np.array(self.leader_target_pos) - np.array(self.leader_agent.pos))

        self.leader_agent.set_vel(0,0)
        self.leader_agent.orientation = np.random.rand()*2*np.pi
        self.leader_agent.done = False
        self.leader_agent.target = False
        observations = self.observe_leader(self.leader_agent, self.leader_target_pos, [0,0], target_distance, 0)
        target_info = self.leader_agent.target

        return observations, self.leader_agent.done

    def step(self, action, num_step,last_distance, last_obs_distacnce):
        """输入leader_action[线速度，角速度]
                        }"""
        
        self._apply_leader_action(self.leader_agent,action, self.leader_target_pos)
        
        observation = self.observe_leader(self.leader_agent, self.leader_target_pos, action, last_distance, num_step)
        reward = self._calculate_leader_reward(agent_id="leader_agent", agent=self.leader_agent, formation_target=self.leader_target_pos, action = action, t=num_step,
                                                last_distance=last_distance, last_obs_distance=last_obs_distacnce)
        
        done = self.leader_agent.done
        target = self.leader_agent.target

        return observation, reward, done, target
    
    def _apply_leader_action(self, agent, action, target):#TODO
        """假设输入的动作是[线速度m/s,角速度 弧度]"""
        linear_vel = action[0] +1
        angular_vel = action[1] * (np.pi/2)
        # new_orientation = agent.orientation + action[1]*self.display_time
        new_orientation = agent.orientation + angular_vel*self.display_time
        new_orientation = new_orientation % (2*np.pi)

        dx = linear_vel * self.display_time * cos(new_orientation)
        dy = linear_vel * self.display_time * sin(new_orientation)
        x = agent.pos[0] + dx
        y = agent.pos[1] + dy
        new_pos = [x,y]

        target_dis  = sqrt((x-target[0])**2 + (y-target[1])**2)
        if target_dis < self.target_radius  :#到达目标点
            agent.target = True

        if x<self.agent_radius or x>self.width - self.agent_radius or y<self.agent_radius or y>self.height - self.agent_radius:
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
            if np.linalg.norm(np.array(new_pos) -np.array(obs_pos)) <= self.obs_radius + self.agent_radius:
                return True
        return False
    
   
    def observe_leader(self, agent, target, action, last_distance, t):
        """领航者自身位置，领航者与目标的距离和角度，与最近障碍物之间的距离（还有啥要加的？TODO）"""
        
        _dis, _angle = CustomEnv.calculate_relative_distance_and_angle(agent.pos, target)

        self_pos_ = [agent.pos[0] / self.width, agent.pos[1] / self.height]
        self_pos_2 = [agent.orientation / (2*np.pi), ((action[0] +1) * cos(agent.orientation))/2, ((action[0] +1) * sin(agent.orientation))/2]
        target_pos_ = [self.leader_target_pos[0] / self.width, self.leader_target_pos[1] / self.height]


        
        obs_distance_angle = []
        obs_pos_vel = []

        for obs_id, obs in self.obstacles.items():
            # obs_pos_ = [obs.pos_x/self.width, obs.pos_y/self.height]
            obs_pos = [obs.pos_x, obs.pos_y]
            _obs_distance, _obs_angle = CustomEnv.calculate_relative_distance_and_angle(agent.pos, obs_pos)

            # obs_dis_threadhold = self.obs_radius *6
            # if _obs_distance < obs_dis_threadhold:
            obs_distance = _obs_distance/self.width
            obs_angle =(_obs_angle - agent.orientation)/(2*np.pi)

            # else :
            #     obs_distance = 1
            #     obs_angle = 0


            
            obs_distance_angle.extend([obs_distance, obs_angle])
        
        for obs in self.obstacles.values():
            px = obs.pos_x / self.width
            py = obs.pos_y / self.height
            vx = 0
            vy = 0
            obs_pos_vel.extend([px, py])
        
       
        
        observation2 = np.array(
            self_pos_ +
            self_pos_2 +
            obs_pos_vel +
            target_pos_ 
            

        )

        return observation2

    
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
        # self.ax.plot(self.leader_agent.pos[0], self.leader_agent.pos[1], 'o', color='purple', markersize=self.agent_radius)
        agent = patches.Circle(self.leader_agent.pos, self.agent_radius, color='purple', fill = True)
        self.ax.add_patch(agent)
        

        # 绘制障碍物
        # for i,obs in enumerate(self.obstacles.values()):
            # self.ax.plot(obs.pos_x, obs.pos_y, 'o', color='red', markersize=self.obs_radius)
        obses = [patches.Circle([obs.pos_x, obs.pos_y], self.obs_radius, color='red', fill=True)for obs in self.obstacles.values()]
        for obs_circle in obses:
            self.ax.add_patch(obs_circle)
        # 绘制目标
        # self.ax.plot(self.leader_target_pos[0], self.leader_target_pos[1], 'o', color='green', markersize=10*10 * np.pi)
        target = patches.Circle(self.leader_target_pos, self.target_radius, color='green', fill=True)
        self.ax.add_patch(target)

        plt.pause(self.display_time)  # 暂停以更新图形
        # plt.show()

    def render_close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None

    

    def _calculate_leader_reward(self, agent_id, agent, formation_target, action, t, last_distance, last_obs_distance):#TODO
        
        reward1 = self._caculate_leader_vo_reward(agent, formation_target, action )
        reward2 = self._caculate_target_reward(agent_id, agent, formation_target, action, t, last_distance )
        reward3 = self._caculate_obstacle_reward(agent_id, agent, last_obs_distance)
        reward4 = self._caculate_velocity_reward(agent, action)
        reward5 = self._caculate_side_reward(agent)
        reward6 = self._caculate_time_reward(t)
        
        # reward = reward1[2] + reward2
        reward =  reward2  + reward3 +reward4  + reward5 
        # if t%500 == 0:
        #     print(f"target_reward : {reward2:.2f} obstacle_reward : {reward3:.2f} velocity_reward :{reward4:.2f} side_reward{reward5}")
        return reward

   
    def _caculate_leader_vo_reward(self, agent, target, agent_action):
        vx = agent.orientation * cos(agent.orientation)
        vy = agent.orientation * sin(agent.orientation)
        robot_state = [agent.pos[0], agent.pos[1], vx, vy, self.agent_radius]
        nei_state_list = []
        obs_cir_list = []
        for obs in self.obstacles.values():
            dis = np.linalg.norm(obs.position() - self.leader_agent.position())
            if dis < self.agent_radius*3:
                obs_state = [obs.pos_x, obs.pos_y, obs.xy_vel[0], obs.xy_vel[1], self.obs_radius]
                obs_cir_list.append(obs_state)
        obs_line_list = []
        action = [vx, vy]
        vo_flag, min_exp_time, min_dis = self.rvo_inter.config_vo_reward(robot_state=robot_state,
                                                                                                                                        nei_state_list=nei_state_list,
                                                                                                                                        obs_cir_list=obs_cir_list,
                                                                                                                                        obs_line_list=obs_line_list,
                                                                                                                                        action=action)
        if vo_flag:
            reward = - min_exp_time
        else:
            reward = 0
        return reward
    
    
    def _caculate_target_reward(self, check_agent_id, check_agent, formation_target, action, t, last_distance):
        """和编队目标之间的距离"""
        dis, angle = CustomEnv.calculate_relative_distance_and_angle(check_agent.pos, formation_target)

        # 设置最小阈值
        min_dis_threshold = self.target_radius/2
        if dis < min_dis_threshold:
            dis = min_dis_threshold

        
        if check_agent.done and check_agent.target:
            return 200
        
        else :
            dis_ = -(dis - last_distance) * 40

            if dis > self.width * 0.3 :
                dis_reward = 0
            # elif self.width * 0.2< dis <=self.width * 0.8:
            #     dis_reward = 1/dis
            elif dis <= self.width * 0.3:
                # dis_reward = 1.125- dis/self.width * 0.2
                dis_reward = min_dis_threshold/(dis + min_dis_threshold)
            
            # reward = dis_
            reward = dis_ + dis_reward/2
            # print(((action[0] + 1) /2)/1.5, - abs(action[1]) /2,dis_  ,dis_reward )
            # print((1- (dis_ / 100))*2)
            return reward 
        # if np.isnan(reward) or np.isinf(reward):
        #     print(f"NaN or Inf detected in reward calculation! reward: {reward}, dis: {dis}, action: {action}")
        #     reward = -100  # 或其他合理的默认值

            

    def _caculate_obstacle_reward(self, agent_id, agent, last_obs_distance):
        reward = 0
        
        for obs_id, obs in self.obstacles.items():
            dis, angle = CustomEnv.calculate_relative_distance_and_angle(agent.pos, [obs.pos_x, obs.pos_y])
            if dis > self.obs_radius * 3:
                x =0
            elif self.obs_radius + self.agent_radius  < dis <= self.obs_radius * 3:
                if agent.done:
                    return -200
                x = -(1/dis - 1/(self.obs_radius * 3))
            elif dis <= self.obs_radius + self.agent_radius :
                return -200
            
            reward += x *100
        # reward_ = reward/self.num_obstacles
            
            # dis, angle = CustomEnv.calculate_relative_distance_and_angle(agent.pos, [obs.pos_x, obs.pos_y])

            # if dis <= self.obs_radius *4: 
            #     # if agent.done and not agent.target:
            #     #     return -150
                
            #     d_dis = dis - last_obs_distance[obs_id]
            #     reward += d_dis * 100
            
        return  reward 
    
    def _caculate_velocity_reward(self, agent, action):
        return ((action[0]+1)/2 - abs(action[1]) ) 

    def _caculate_side_reward(self, agent):
        reward = 0
        distances = [agent.pos[0], self.width - agent.pos[0], agent.pos[1], self.height - agent.pos[1]]
        
        for dis in distances:
            if dis > self.width * 0.18:
                re = 0
            
            elif self.width *0.05 < dis <= self.width* 0.18:
                re = -self.width *0.05 /dis
            elif dis <= self.width *0.05:
                if agent.done and not agent.target:
                    return -100
                re = -20 *((self.width *0.05)/(self.width *0.05 + dis))

            reward += re

        return reward *2

    def _caculate_time_reward(self, t):

        return -t/2000


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
        if angle < 0:
            angle = angle + 2*np.pi
        
        
        return distance, angle