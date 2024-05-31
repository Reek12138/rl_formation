# -*- coding: utf-8 -*-
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils import agent_selector
from gym.spaces import Box, Discrete
import numpy as np
from pettingzoo.mpe import simple_adversary_v3
from .obstacle import obstacle
from .circle_agent import circle_agent
import matplotlib.pyplot as plt
from math import sin, cos, tan, pi, sqrt
from .rvo_inter import rvo_inter



LEADER_MAX_LINEAR_VEL = 4

class CustomEnv:
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents=4, width=300, height=300, num_obstacles=15, agent_radius=5, safe_theta = 2,
                 target_pos = np.array([275, 275]), delta = 0.1):
        super().__init__()
        self.num_agents = num_agents
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.obs_radius = 5
        self.agent_radius = agent_radius
        self.fig = None
        self.ax = None
        self.formation_pos = {"agent_0":[-15,-15],
                              "agent_1":[-15,15],
                              "agent_2":[15,-15],
                              "agent_3":[15,15]}
        self.display_time = delta
        self.safe_theta = safe_theta
        self.rvo_inter = rvo_inter()

        # 生成circle_agent实例列表
        self.leader_agent = circle_agent(self.agent_radius, pos_x=50, pos_y=50, n_agent=1, 
                                        #  obs_dim=4+2*self.num_obstacles,
                                         obs_dim=10,
                                        #  state_dim=4+2*self.num_obstacles,
                                         state_dim=10,
                                         memo_size=100000,
                                         action_dim=2)

        self.follower_state_dim = 2 + 2 + 2*(self.num_agents-1) + 2*self.num_obstacles
        self.follower_agents = {
            f"agent_{i}": circle_agent(
                radius=agent_radius,
                pos_x=self.leader_agent.pos_x + i*20 +np.random.rand() * 10,
                pos_y=self.leader_agent.pos_y + i*20 +np.random.rand() * 10,
                memo_size=100000,
                obs_dim=self.follower_state_dim ,
                state_dim=self.follower_state_dim* (self.num_agents ),
                action_dim=2
                ) 
                for i in range(self.num_agents)}

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
        #    重置位置
        # self.leader_agent.set_position(5,5)
        # for i,agent in enumerate(self.follower_agents.values()) :
        #     agent.set_position(self.leader_agent.pos_x + i*3 +3, self.leader_agent.pos_y + i*3 +3)
        
        # self.leader_agent.set_position(25,25)

        self.leader_agent.set_position(10+np.random.rand()*self.width*0.8, 10+np.random.rand()*self.height*0.8)
        self.leader_target_pos = [10+np.random.rand()*self.width*0.8, 10+np.random.rand()*self.height*0.8]

        self.leader_agent.set_xy_vel(0,0)
        self.leader_agent.set_linear_orientation(0,0)
        self.leader_agent.orientation = np.random.rand()*2*np.pi
        self.leader_agent.done = False
        self.leader_agent.target = False
        # for i,agent in enumerate(self.follower_agents.values()):
        #     agent.set_position(self.leader_agent.pos_x + self.formation_pos[i][0]*15 + np.random.rand() * 5,
        #                         self.leader_agent.pos_y + self.formation_pos[i][1]*15 + np.random.rand() * 5)
        for agent_id, agent in self.follower_agents.items():
            agent.set_position(self.leader_agent.pos_x + self.formation_pos[agent_id][0] + np.random.rand()*5,
                               self.leader_agent.pos_y + self.formation_pos[agent_id][1] + np.random.rand()*5)
            agent.set_xy_vel(0,0)
            agent.set_linear_orientation(0,0)
            agent.orientation = 0
            agent.done = False
            agent.target = False
            

    
        observations = {}
        rewards = {}
        dones = {}
        target_infos = {}

        observations["leader_agent"] = self.observe_leader(self.leader_agent, self.leader_target_pos, [-1,0])
        # rewards["leader_agent"] = self._calculate_leader_reward(agent_id="leader_agent", agent=self.leader_agent, formation_target=self.leader_target_pos, action = leader_action)
        dones["leader_agent"] = self.leader_agent.done
        target_infos["leader_agent"] = self.leader_agent.target

        for agent_id ,agent in self.follower_agents.items():
            target_x = self.leader_agent.pos_x + self.formation_pos[agent_id][0]
            target_y = self.leader_agent.pos_y + self.formation_pos[agent_id][1]
            formation_target = [target_x, target_y]

            observations[agent_id] = self.observe(agent_id, agent, formation_target)
            
            # print(agent_id,observations[agent_id])
            # rewards[agent_id] = self._calculate_follower_reward(agent_id, agent, formation_target, follower_actions[agent_id])
            dones[agent_id] = agent.done
            target_infos[agent_id] = agent.info

        return observations, target_infos

    def step(self, leader_action, follower_actions):
        """输入leader_action[线速度，角速度]
                        follower_actions = {"agent_0": action,
                                                                 "agent_1": action,
                                                                 "agent_2": action,
                                                                 "agent_3": action }"""
        
        self._apply_leader_action(self.leader_agent, leader_action, self.leader_target_pos)

        for agent_id,action in follower_actions.items():
            target_x = self.leader_agent.pos_x + self.formation_pos[agent_id][0]
            target_y = self.leader_agent.pos_y + self.formation_pos[agent_id][1]
            formation_target = [target_x, target_y]
            self._apply_follower_action(self.follower_agents[agent_id], action, formation_target)

        for agent_id, agent in self.follower_agents.items():
            self._check_follower_agent_collision(agent_id, agent)
        
        observations = {}
        rewards = {}
        dones = {}
        target_infos = {}

        observations["leader_agent"] = self.observe_leader(self.leader_agent, self.leader_target_pos, leader_action)
        rewards["leader_agent"] = self._calculate_leader_reward(agent_id="leader_agent", agent=self.leader_agent, formation_target=self.leader_target_pos, action = leader_action)
        dones["leader_agent"] = self.leader_agent.done
        target_infos["leader_agent"] = self.leader_agent.target

        for agent_id ,agent in self.follower_agents.items():
            target_x = self.leader_agent.pos_x + self.formation_pos[agent_id][0]
            target_y = self.leader_agent.pos_y + self.formation_pos[agent_id][1]
            formation_target = [target_x, target_y]

            observations[agent_id] = self.observe(agent_id, agent, formation_target)
            rewards[agent_id] = self._calculate_follower_reward(agent_id, agent, formation_target, follower_actions[agent_id])
            dones[agent_id] = agent.done
            target_infos[agent_id] = agent.target

        return observations, rewards, dones, target_infos
    
    def _apply_leader_action(self, agent, action, target):
        """假设输入的动作是[线速度m/s,角速度 弧度]"""
        linear_vel = ((action[0]+1)/2) * LEADER_MAX_LINEAR_VEL
        # linear_vel = action[0] 
        angular_vel = -(action[1]) * (np.pi/2)
        # new_orientation = agent.orientation + action[1]*self.display_time
        new_orientation = agent.orientation + angular_vel*self.display_time
        new_orientation % (2*np.pi)

        dx = linear_vel * self.display_time * cos(new_orientation)
        dy = linear_vel * self.display_time * sin(new_orientation)
        x = agent.pos_x + dx
        y = agent.pos_y + dy
        new_pos = [x,y]

        target_dis  = sqrt((x-target[0])**2 + (y-target[1])**2)
        if target_dis < 3:
            agent.target = True

        if x<0 or x>300 or y<0 or y>300:
            flag = True
        else:
            flag = False

        # if not self._check_obs_collision(agent, new_pos)and not flag:
        if not flag:
            agent.set_position(x, y)  
            agent.set_linear_orientation(action[0], action[1])
            agent.orientation = new_orientation
        else:
            agent.done = True

    def _apply_follower_action(self, agent, action, target):
        """ 假设输入的是xy轴的速度"""
        dx = action[0] * self.display_time 
        dy = action[1] * self.display_time
        x = agent.pos_x + dx
        y = agent.pos_y + dy
        new_pos = [x,y]

        target_dis  = sqrt((x-target[0])**2 + (y-target[1])**2)
        if target_dis < 3:
            agent.target = True             

        if not self._check_obs_collision(agent, new_pos):
            agent.set_position(x, y)  
            agent.set_xy_vel(action[0], action[1])
        else:
            agent.done = True

    def _check_obs_collision(self,current_agent,new_pos):
        """检查智能体是否与障碍物碰撞"""
        for obs in self.obstacles.values():
            obs_pos = [obs.pos_x, obs.pos_y]
            if np.linalg.norm(np.array(new_pos) -np.array(obs_pos)) < self.obs_radius:
                return True
        return False
    
    def _check_follower_agent_collision(self, check_agent_id, check_agent):
        """检查agent之间有无碰撞"""
        for agent_id, agent in self.follower_agents.items():
            if agent_id != check_agent_id:
                dx = agent.pos_x - check_agent.pos_x
                dy = agent.pos_y - check_agent.pos_y
                dis = sqrt(dx**2 + dy**2)
                if dis < 2 * self.agent_radius + self.safe_theta:
                    check_agent.done = True

    def observe_leader(self, agent, target, action):
        """领航者自身位置，领航者与目标的距离和角度，与最近障碍物之间的距离（还有啥要加的？TODO）"""
        self_pos = [agent.pos_x, agent.pos_y]
        target_dis, target_angle = CustomEnv.calculate_relative_distance_and_angle(self_pos, target)
        xy_dis = [agent.pos_x-0, 300-agent.pos_x, agent.pos_y-0, 300-agent.pos_y]
        action[0] = (action[0] + 1)/2
        action[1] = action[1] * np.pi
        
        obs_distance_angle = []

        for obs_id, obs in self.obstacles.items():
            obs_pos = [obs.pos_x, obs.pos_y]
            obs_distance, obs_angle = CustomEnv.calculate_relative_distance_and_angle(self_pos, obs_pos)
            obs_distance_angle.extend([obs_distance, obs_angle])
        
        # observation = np.array(self_pos + [target_dis, target_angle] + obs_distance_angle)
        observation = np.array(self_pos + [target_dis, target_angle] + xy_dis + [action[0], action[1]])

        return observation

    def observe(self, agent_id, agent, target):
        """
        agent自身位置,与目标的距离和角度,与最近障碍物的距离和角度,与集群中其他智能体的距离和角度
        observation =[self_pos_x, self_pos_y, targetpos_x, targetpos_y, obs1_dis, obs1_ang,....... agent1_dis,agent1_ang....]
        """
        self_pos = [agent.pos_x, agent.pos_y]
        target_dis, target_angle = CustomEnv.calculate_relative_distance_and_angle(self_pos, target)

        other_agents_distance_angle = []

        for follower_agent_id, follower_agent in self.follower_agents.items():
            if follower_agent_id != agent_id:
                other_pos = [follower_agent.pos_x, follower_agent.pos_y]
                distance, angle = CustomEnv.calculate_relative_distance_and_angle(self_pos, other_pos)
                other_agents_distance_angle.extend([distance, angle])
        
        # closest_obs_pos = self._find_closest_obstacle(self_pos)
        # obs_distance, obs_angle = self.calculate_relative_distance_and_angle(self_pos, closest_obs_pos)
        obs_distance_angle = []

        for obs_id, obs in self.obstacles.items():
            obs_pos = [obs.pos_x, obs.pos_y]
            obs_distance, obs_angle = CustomEnv.calculate_relative_distance_and_angle(self_pos, obs_pos)
            obs_distance_angle.extend([obs_distance, obs_angle])

        # observation = np.array([target_dis, target_angle, obs_distance, obs_angle] + other_agents_distance_angle)
        observation = np.array(self_pos + [target_dis, target_angle] + obs_distance_angle + other_agents_distance_angle)

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
        self.ax.plot(self.leader_agent.pos_x, self.leader_agent.pos_y, 'o', color='purple', markersize=self.agent_radius)
        for agent in self.follower_agents.values():
            self.ax.plot(agent.pos_x, agent.pos_y, 'o', color='blue', markersize=self.agent_radius)

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

    

    # Check if the agent has reached the target
    # def _is_done(self, agent_id, agent):
    #     if self.dones[agent] == True:
    #         return True
    #     elif self.dones[agent] == False:
    #         distance2target = np.linalg.norm(self.agent_pos[agent] - self.target_pos[agent])
    #         return distance2target < self.agent_pos + 2

    def _calculate_leader_reward(self, agent_id, agent, formation_target, action ):#TODO
        
        reward1 = self._caculate_leader_vo_reward(agent, formation_target, action )
        reward2 = self._caculate_formation_reward(agent_id, agent, formation_target, action,  "leader" )
        
        # reward = reward1[2] + reward2
        reward =  reward2 
        return reward

    def _calculate_follower_reward(self, check_agent_id, check_agent, formation_target, agent_action):#TODO
        # Calculate the reward for the agent
        
        reward1 =  self._caculate_follower_vo_reward(check_agent_id, check_agent, formation_target, agent_action)
        reward2 = self._caculate_formation_reward(check_agent_id, check_agent, formation_target, agent_action, "follower")
        # reward = reward1[2] + reward2
        reward =  reward2
        return reward
    
    def _caculate_leader_vo_reward(self, agent, target, agent_action):
        vx = agent.linear_orientation[0] * cos(agent.orientation)
        vy = agent.linear_orientation[0] * sin(agent.orientation)
        robot_state = [agent.pos_x, agent.pos_y, vx, vy, self.agent_radius]
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
    
    def _caculate_follower_vo_reward(self, check_agent_id, check_agent, formation_target, agent_action):
        """
        vo类信息
        reward = [vo_flag, min_exp_time, min_dis]
        """
        robot_state = [check_agent.pos_x, check_agent.pos_y, check_agent.xy_vel[0], check_agent.xy_vel[1], self.agent_radius]
        nei_state_list = []
        for agent_id, agent in self.follower_agents.items():
            if agent_id != check_agent_id:
                other_robot_state = [agent.pos_x, agent.pos_y, agent.xy_vel[0], agent.xy_vel[1], self.agent_radius]
                nei_state_list.append(other_robot_state)

        obs_cir_list = []
        for obs in self.obstacles.values():
            obs_state = [obs.pos_x, obs.pos_y, obs.xy_vel[0], obs.xy_vel[1], self.obs_radius]
            obs_cir_list.append(obs_state)

        obs_line_list = []
        action = agent_action

        vo_flag, min_exp_time, min_dis = self.rvo_inter.config_vo_reward(robot_state=robot_state,
                                                                                                                                        nei_state_list=nei_state_list,
                                                                                                                                        obs_cir_list=obs_cir_list,
                                                                                                                                        obs_line_list=obs_line_list,
                                                                                                                                        action=action)
        reward = [vo_flag, min_exp_time, min_dis]
        return reward

    def _caculate_formation_reward(self, check_agent_id, check_agent, formation_target, action, agent_type):
        """和编队目标之间的距离"""
        # dx = check_agent.pos_x - formation_target[0]
        # dy = check_agent.pos_y - formation_target[1]
        # dis = sqrt(dx**2 + dy**2)
        dis, angle = CustomEnv.calculate_relative_distance_and_angle([check_agent.pos_x, check_agent.pos_y], formation_target)

        

        if agent_type == "follower":
            reward = - dis/3 #TODO要不要考虑归一化
        
        if agent_type == "leader":

            if check_agent.done:
                return -100
        
            elif check_agent.target:
                return 200
            # reward = - (dis / self.width) *1.2
            else:
                dis_ = dis/1.414
            # if dis_ >= 250:
            #     extra_reward = 0
            # elif 200 < dis_ <= 250 :
            #     extra_reward = 20
            # elif 150 < dis_ <= 200:
            #     extra_reward = 30
            # elif 100 < dis_ <= 150:
            #     extra_reward = 40
            # elif 50 < dis_ <= 100:
            #     extra_reward = 50
            # elif dis_ <= 50:
            #     extra_reward = 60

            
                reward = ((action[0]+1)/2)*10 -abs(action[1]*10) - abs(angle)*10 + (300 - dis/1.414)/30
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