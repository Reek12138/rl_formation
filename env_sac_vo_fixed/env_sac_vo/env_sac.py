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

    def __init__(self, width=100, height=100, num_obstacles=15, agent_radius=1, obs_radius = 2,safe_theta = 8,target_radius = 4,
                 target_pos = np.array([50, 50]), delta = 0.1, memo_size=100000):
        super().__init__()
        
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.obs_radius = obs_radius
        self.target_radius = target_radius
        self.agent_radius = agent_radius
        self.obs_delta =8
        self.fig = None
        self.ax = None
        
        self.display_time = delta
        self.safe_theta = safe_theta
        self.rvo_inter = rvo_inter()

        self.agent_trajectory = []
        self.last_action = [0, 0]

        

        # 生成circle_agent实例列表
        self.leader_agent = circle_agent(self, pos=[25, 25], vel=[0,0], orientation=0, memo_size=memo_size,
                                         state_dim=11 + self.num_obstacles * 5,
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

        # 假设前9个障碍物是固定的
        fixed_obstacles = obstacles_list[:9]
        random_obstacles = obstacles_list[9:]

        for i, obs in enumerate(random_obstacles):
            key = obstacles_keys[9 + i]
            is_position_valid = False

            while not is_position_valid:
                is_position_valid = True

                # 仅检查与之前的随机障碍物的距离
                for j in range(i):
                    obs2 = random_obstacles[j]
                    dis = np.linalg.norm(self.obstacles[key].position() - obs2.position())
                    if dis < 2 * self.obs_radius + self.agent_radius + self.safe_theta:
                        is_position_valid = False
                        break

                # 检查与固定障碍物的距离
                for fixed_obs in fixed_obstacles:
                    dis_fixed = np.linalg.norm(self.obstacles[key].position() - fixed_obs.position())
                    if dis_fixed < 2 * self.obs_radius + self.agent_radius + self.safe_theta:
                        is_position_valid = False
                        break

                # 检查与目标位置的距离
                dis2 = np.linalg.norm(np.array(self.leader_target_pos) - self.obstacles[key].position())
                if dis2 < self.obs_radius + self.target_radius + self.agent_radius + self.safe_theta:
                    is_position_valid = False

                # 如果位置无效，则重新生成随机位置
                if not is_position_valid:
                    self.obstacles[key].pos_x = np.random.rand() * self.width * 0.7 + self.width * 0.15
                    self.obstacles[key].pos_y = np.random.rand() * self.height * 0.7 + self.height * 0.15


    def _check_obs_agent(self, agent):
        # obstacles_keys = list(self.obstacles.keys())
        # obstacles_list = list(self.obstacles.values())

        # # 假设前9个障碍物是固定的
        # fixed_obstacles = obstacles_list[:9]
        # random_obstacles = obstacles_list[9:]
        for obs in self.obstacles.values():
            dis = np.linalg.norm(obs.position() - agent.position())
            if dis <= self.obs_radius + self.agent_radius + self.safe_theta/2:
                return True
        return False
    
    def _check_obs_target(self, target_pos):
        obstacles_keys = list(self.obstacles.keys())
        obstacles_list = list(self.obstacles.values())

        # 假设前9个障碍物是固定的
        fixed_obstacles = obstacles_list[:9]
        random_obstacles = obstacles_list[9:]

        for obs in fixed_obstacles:
            dis = np.linalg.norm(obs.position() - np.array(target_pos))
            if dis < self.obs_radius + self.target_radius + self.safe_theta/2:
                return True
        return False

    def reset(self):


        # 随机位置的障碍物
        for i in range(len(self.fix_position), self.num_obstacles):
            self.obstacles[f"obstacle_{i}"].pos_x = np.random.rand() * self.width * 0.7 + self.width * 0.15
            self.obstacles[f"obstacle_{i}"].pos_y=np.random.rand() * self.height * 0.7 + self.height * 0.15
            
        
        

        self.leader_target_pos = [self.width*0.1+np.random.rand()*self.width*0.8, self.height*0.1+np.random.rand()*self.height*0.8]

        self.leader_agent.set_position(self.width*0.10+np.random.rand()*self.width*0.8, self.height*0.10+np.random.rand()*self.height*0.8)
        
        

        flag2 = self._check_obs_target(self.leader_target_pos)
        while flag2:
            self.leader_target_pos = [self.width*0.1+np.random.rand()*self.width*0.8, self.height*0.1+np.random.rand()*self.height*0.8]
            flag2 = self._check_obs_target(self.leader_target_pos)

        self._check_obs()
        
        flag1 = self._check_obs_agent(self.leader_agent)
        while  np.linalg.norm(np.array(self.leader_target_pos) - np.array(self.leader_agent.pos)) < self.agent_radius + self.target_radius + self.safe_theta*3 or flag1:
            self.leader_agent.set_position(self.width*0.10+np.random.rand()*self.width*0.8, self.height*0.10+np.random.rand()*self.height*0.8)
            flag1 = self._check_obs_agent(self.leader_agent)

        target_distance =  np.linalg.norm(np.array(self.leader_target_pos) - np.array(self.leader_agent.pos))

        

        self.leader_agent.set_vel(0)
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
            agent.vel = linear_vel
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

        side_pos = [agent.pos[0] / self.width, (self.width - agent.pos[0]) / self.width, agent.pos[1] / self.height, (self.height - agent.pos[1]) / self.height]
        target_pos_2 = [(self.leader_target_pos[0] - agent.pos[0]) / self.width, (self.leader_target_pos[1] - agent.pos[1]) / self.height, _dis / (self.width * 1.414), (_angle - agent.orientation) / (2 * np.pi)]

        
        obs_distance_angle = []
        obs_pos_vel = []

        for obs_id, obs in self.obstacles.items():
            obs_pos = [obs.pos_x, obs.pos_y]
            _obs_distance, _obs_angle = CustomEnv.calculate_relative_distance_and_angle(agent.pos, obs_pos)

            
            # obs_distance_angle.extend([obs_distance, obs_angle])
            if _obs_distance <= self.obs_delta:
                vx = agent.vel * cos(agent.orientation)
                vy = agent.vel * sin(agent.orientation)
                robot_state = [agent.pos[0], agent.pos[1], vx, vy, self.agent_radius]
                nei_state_list = []
                obs_cir_list = [ [obs.pos_x, obs.pos_y, obs.xy_vel[0], obs.xy_vel[1], self.obs_radius * 1.2]]#放大
                obs_line_list = []
                action = [vx, vy]
                vo_flag, min_exp_time, min_dis = self.rvo_inter.config_vo_reward(robot_state=robot_state,
                                                                                                                                                        nei_state_list=nei_state_list,
                                                                                                                                                        obs_cir_list=obs_cir_list,
                                                                                                                                                        obs_line_list=obs_line_list,
                                                                                                                                                        action=action)
                # print("vo_flag :" ,vo_flag)
                px = (obs.pos_x - agent.pos[0]) / self.obs_delta
                py = (obs.pos_y - agent.pos[1])/ self.obs_delta
                vx = 0
                vy = 0
                _obs_distance_ = _obs_distance / (self.obs_delta * 1.415)
            
            else:
                vo_flag = False
                px = 0
                py = 0
                _obs_distance_ = 0

                # px = 1
                # py = 1
                # _obs_distance_ = 1
            
            obs_dis_angle = (_obs_angle - agent.orientation)
            obs_pos_vel.extend([px, py, _obs_distance_, obs_dis_angle / (2*np.pi), vo_flag])
            # obs_pos_vel.extend([px, py, vo_flag])
     
        observation2 = np.array(
            self_pos_ +
            self_pos_2 +
            obs_pos_vel+
            target_pos_ 
            

        )
        observation1 = np.array(
            side_pos + 
            target_pos_2 +
            obs_pos_vel +
            # self_pos_ +
            self_pos_2
        )
        

        return observation1

    
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

        arrow_length = self.agent_radius * 1  # Adjust length as needed
        arrow_dx = arrow_length * np.cos(self.leader_agent.orientation)
        arrow_dy = arrow_length * np.sin(self.leader_agent.orientation)
        arrow = patches.FancyArrow(
            self.leader_agent.pos[0], 
            self.leader_agent.pos[1], 
            arrow_dx, 
            arrow_dy, 
            width=self.agent_radius * 0.25, 
            color='purple'
        )
        self.ax.add_patch(arrow)

        # 记录智能体当前的位置到轨迹
        self.agent_trajectory.append(self.leader_agent.pos.copy())

        # 绘制智能体的轨迹
        if len(self.agent_trajectory) > 1:
            traj_x, traj_y = zip(*self.agent_trajectory)
            self.ax.plot(traj_x, traj_y, color='blue', linestyle='--', marker='o', markersize=1, label='Trajectory')

        

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
        reward ,reward1, reward2,reward3, reward4, reward5 = 0,0,0,0,0,0
        vo_flag, reward1, min_dis = self._caculate_leader_vo_reward(agent, formation_target, action )
        reward2 = self._caculate_target_reward(agent_id, agent, formation_target, action, t, last_distance ,vo_flag)
        reward3 = self._caculate_obstacle_reward(agent_id, agent, last_obs_distance)
        reward4 = self._caculate_velocity_reward(agent, action, vo_flag, self.last_action)
        reward5 = self._caculate_side_reward(agent)
        reward6 = self._caculate_time_reward(t)
        
        # reward = reward1[2] + reward2
        # if min_dis + self.obs_radius <= self.obs_delta :
        reward = reward2 + reward3 + reward4 + reward5

        self.last_action = action
        # else:
        #     reward = reward2 + reward4 + reward5 
        # if reward1 != 0:
        #     print(f"vo_reward : {reward1}")
        # if t % 10 == 0:
            # print(f"vo_flag : {vo_flag}, target_reward : {reward2:.2f}, obs_reward : {reward3:.2f}, velocity_reward : { reward4:.2f}, side_reward:{reward5:.2f}, action :{action}")
        return reward

   
    def _caculate_leader_vo_reward(self, agent, target, agent_action):
        vx = agent.vel * cos(agent.orientation)
        vy = agent.vel * sin(agent.orientation)
        robot_state = [agent.pos[0], agent.pos[1], vx, vy, self.agent_radius]
        nei_state_list = []
        obs_cir_list = []
        for obs in self.obstacles.values():
            dis = np.linalg.norm(obs.position() - self.leader_agent.position())
            if dis <= self.obs_delta:
                obs_state = [obs.pos_x, obs.pos_y, obs.xy_vel[0], obs.xy_vel[1], self.obs_radius * 1.2]#放大
                obs_cir_list.append(obs_state)
        # obs_line_list = [ np.array([[0, 0], [0, 100]]) ,
        #                     np.array([[0, 100], [100, 100]]) ,
        #                     np.array([[100, 100], [100, 0]]) , 
        #                     np.array([[100, 0], [0, 0]]) ]
        obs_line_list = []
        action = [vx, vy]
        vo_flag, min_exp_time, min_dis = self.rvo_inter.config_vo_reward(robot_state=robot_state,
                                                                                                                                        nei_state_list=nei_state_list,
                                                                                                                                        obs_cir_list=obs_cir_list,
                                                                                                                                        obs_line_list=obs_line_list,
                                                                                                                                        action=action)
        if vo_flag :
            # reward = - 10/(min_exp_time + 0.1)
            reward = - 2
        else:
            reward = 0
        return vo_flag, reward *2, min_dis


    def _caculate_target_reward(self, check_agent_id, check_agent, formation_target, action, t, last_distance, vo_flag):
        """和编队目标之间的距离"""
        reward = 0
        dis, angle = CustomEnv.calculate_relative_distance_and_angle(check_agent.pos, formation_target)

        # 设置最小阈值
        min_dis_threshold = self.target_radius/2
        if dis < min_dis_threshold:
            dis = min_dis_threshold

        
        if check_agent.done and check_agent.target:
            return 500
        
        else :
            dis_ = -(dis - last_distance) 

            if dis > self.width * 0.3 :
                dis_reward = 0
            # elif self.width * 0.2< dis <=self.width * 0.8:
            #     dis_reward = 1/dis
            elif dis <= self.width * 0.3:
                # dis_reward = 1.125- dis/self.width * 0.2
                dis_reward = min_dis_threshold/(dis + min_dis_threshold)
            
            reward = dis_
            # reward = dis_ + dis_reward/2
            # print(((action[0] + 1) /2)/1.5, - abs(action[1]) /2,dis_  ,dis_reward )
            # print((1- (dis_ / 100))*2)
            if vo_flag:
                # return 0
                # return reward *100
                return reward *150

            else:
                # return reward *500
                return reward *700
        # if np.isnan(reward) or np.isinf(reward):
        #     print(f"NaN or Inf detected in reward calculation! reward: {reward}, dis: {dis}, action: {action}")
        #     reward = -100  # 或其他合理的默认值

            

    def _caculate_obstacle_reward(self, agent_id, agent, last_obs_distance):
        if agent.done and not agent.target:
                    return -500
        
        reward = 0
        
        for obs_id, obs in self.obstacles.items():
            dis, angle = CustomEnv.calculate_relative_distance_and_angle(agent.pos, [obs.pos_x, obs.pos_y])
            if dis <= self.obs_delta and abs(angle - agent.orientation)<=(np.pi/2)*1.2:
                vx = agent.vel * cos(agent.orientation)
                vy = agent.vel * sin(agent.orientation)
                robot_state = [agent.pos[0], agent.pos[1], vx, vy, self.agent_radius]
                nei_state_list = []
                obs_cir_list = [ [obs.pos_x, obs.pos_y, obs.xy_vel[0], obs.xy_vel[1], self.obs_radius * 1.2]]#放大
                obs_line_list = []
                action = [vx, vy]
                vo_flag, min_exp_time, min_dis = self.rvo_inter.config_vo_reward(robot_state=robot_state,
                                                                                                                                                        nei_state_list=nei_state_list,
                                                                                                                                                        obs_cir_list=obs_cir_list,
                                                                                                                                                        obs_line_list=obs_line_list,
                                                                                                                                                        action=action)
                if vo_flag:
                    delta = 800
                    x = 60
                    # x = 0
                    # x = (dis - self.obs_delta)
                    # relative_pos = obs.position() - agent.position()
                    # # 计算相对位置的单位向量
                    # relative_pos_unit = relative_pos / np.linalg.norm(relative_pos)
                    # relative_pos = np.squeeze(relative_pos)
                    
                    # angle =self.calculate_angle_between_vectors(np.array([vx, vy]), relative_pos)


                else:
                    delta = 100
                    # x = 20
                    x = 0
                    # angle = 0

                d_dis = dis - last_obs_distance[obs_id]

                # if dis <= self.obs_delta :
                #     x = -(1/dis)
                # else:
                #     x = 0


                # if dis < 5:
                # x = -(1/dis)
                # else:
                #     x = 0
                
                        # print( f"d_dis : {d_dis}, x : {x}")
                    
                reward += d_dis * delta+ -(1/dis) * x
                # reward += d_dis * delta 
                # reward +=  x * 50
            
        return  reward 
    
    def _caculate_velocity_reward(self, agent, action, vo_flag, last_action):
        if vo_flag:
            # return ((action[0] + 1) + abs(action[1])/2) *5
            return (action[0] + 1) *5 - abs(action[1]) *2.5  - abs(action[0] - last_action[0])*5
        else:
            # return ((action[0] + 1) ) *5 + (abs(action[1])/2)*3
            return ((action[0] + 1) ) *5 - abs(action[1]) *2.5 - abs(action[0] - last_action[0])*5
        

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

        return reward *3

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
    
    @staticmethod
    def calculate_angle_between_vectors(v1, v2):
        """
        计算两个向量之间的夹角大小（以弧度为单位）。
        
        参数:
        v1: 第一个向量 (vx1, vy1)
        v2: 第二个向量 (vx2, vy2)
        
        返回:
        float: 两个向量之间的夹角（以弧度为单位）
        """
        # 计算点积
        dot_product = np.dot(v1, v2)
        
        # 计算两个向量的模长
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # 计算夹角的余弦值
        cos_theta = dot_product / (norm_v1 * norm_v2)
        
        # 确保cos_theta在[-1, 1]范围内，避免数值误差
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        # 计算夹角，以弧度为单位
        angle = np.arccos(cos_theta)
        
        return angle