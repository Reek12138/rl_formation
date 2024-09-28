# -*- coding: utf-8 -*-
from .rvo_inter import rvo_inter
import numpy as np
from .obstacle import obstacle
from .circle_agent_sac import circle_agent
from .follower_uav import follower_uav
import matplotlib.pyplot as plt
from math import sin, cos, tan, pi, sqrt, log
import matplotlib.patches as patches

class CustomEnv:
    metadata = {'render.modes':['human']}

    def __init__(self, width=100, height=100, num_obstacles=15, agent_radius=1, obs_radius = 2,safe_theta = 8,target_radius = 4,
                 target_pos = np.array([50, 50]), delta = 0.1, memo_size=100000, follower_uav_num = 3):
        super().__init__()

        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.obs_radius = obs_radius
        self.target_radius = target_radius
        self.agent_radius = agent_radius
        self.obs_delta =10
        self.fig = None
        self.ax = None
        
        self.display_time = delta
        self.safe_theta = safe_theta
        self.rvo_inter = rvo_inter()
        self.follower_uav_num = follower_uav_num

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
        
        self.follower_uavs = {}

        self.formation_pos = [
            (0, 2),
            (-1.732, -1),
            (1.732, -1)
        ]

        for i in range (follower_uav_num):
            self.follower_uavs[f"follower_{i}"] = follower_uav(radius=self.agent_radius, pos = self.leader_agent.pos + self.formation_pos[i], vel=[0,0],
                                                               memo_size=100000, state_dim=40, action_dim=2, alpha=1e-4, beta=1e-4,
                                                               alpha_lr=1e-4, hidden_dim=512, gamma=0.99, tau=0.01, batch_size=512,target_entropy=-log(2))
        

        
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
    
    def _check_fix_obs_agent(self, leader_pos):
        obstacles_keys = list(self.obstacles.keys())
        obstacles_list = list(self.obstacles.values())

        # 假设前9个障碍物是固定的
        fixed_obstacles = obstacles_list[:9]
        random_obstacles = obstacles_list[9:]

        for obs in fixed_obstacles:
            if np.linalg.norm(obs.position() - np.array(leader_pos)) < self.agent_radius + self.obs_radius +self.safe_theta/4:
                return True
            
            for i in range(self.follower_uav_num):
                if np.linalg.norm(obs.position() - np.array(self.follower_uavs[f"follower_{i}"].position())) < self.agent_radius +self.obs_radius + self.safe_theta/8:
                    return True
                
            return False


    
    def reset(self):


        # 随机位置的障碍物
        for i in range(len(self.fix_position), self.num_obstacles):
            self.obstacles[f"obstacle_{i}"].pos_x = np.random.rand() * self.width * 0.7 + self.width * 0.15
            self.obstacles[f"obstacle_{i}"].pos_y=np.random.rand() * self.height * 0.7 + self.height * 0.15
            
        
        

        self.leader_target_pos = [self.width*0.1+np.random.rand()*self.width*0.8, self.height*0.1+np.random.rand()*self.height*0.8]

        self.leader_agent.set_position(self.width*0.10+np.random.rand()*self.width*0.8, self.height*0.10+np.random.rand()*self.height*0.8)
        for i in range(self.follower_uav_num):
            self.follower_uavs[f"follower_{i}"].set_position(self.leader_agent.pos[0] + self.formation_pos[i][0], self.leader_agent.pos[1] + self.formation_pos[i][1])

        # 确保开始不与固定障碍物碰撞
        flag0 = self._check_fix_obs_agent(self, self.leader_agent.pos)
        while flag0:
            self.leader_agent.set_position(self.width*0.10+np.random.rand()*self.width*0.8, self.height*0.10+np.random.rand()*self.height*0.8)
            for i in range(self.follower_uav_num):
                self.follower_uavs[f"follower_{i}"].set_position(self.leader_agent.pos[0] + self.formation_pos[i][0], self.leader_agent.pos[1] + self.formation_pos[i][1])
            flag0 = self._check_fix_obs_agent(self, self.leader_agent.pos)

        
        
        # 确保目标不与固定障碍物碰撞
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

        for i in range(self.follower_uav_num):
            self.follower_uavs[f"follower_{i}"].set_vel(0, 0)
            self.follower_uavs[f"follower_{i}"].target = False
            self.follower_uavs[f"follower_{i}"].done = False

        self.leader_agent.done = False
        self.leader_agent.target = False
        observations = self.observe()
        target_info = self.leader_agent.target

        return observations, self.leader_agent.done
    
    def observe(self ):
        """
        输出领航者的观测， 更新跟随者的观测
        跟随者的观测和actor网络都写在自己的节点内
        领航者的观测则作为返回值，跟之前的训练模式一样

        """
        # 领航者的观测更新
        _dis, _angle = CustomEnv.calculate_relative_distance_and_angle(self.leader_agent.pos, self.leader_target_pos)
        side_pos = [self.leader_agent.pos[0] / self.width, (self.width - self.leader_agent.pos[0]) / self.width, self.leader_agent.pos[1] / self.height, (self.height - self.leader_agent.pos[1]) / self.height]
        target_pos_ = [(self.leader_target_pos[0] - self.leader_agent.pos[0]) / self.width, (self.leader_target_pos[1] - self.leader_agent.pos[1]) / self.height, _dis / (self.width * 1.414), (_angle - self.leader_agent.orientation) / (2 * np.pi)]
        self_pos_ = [self.leader_agent.orientation / (2*np.pi), ((action[0] +1) * cos(self.leader_agent.orientation))/2, ((action[0] +1) * sin(self.leader_agent.orientation))/2]
        
        obs_distance_angle = []
        obs_pos_vel = []

        for obs_id, obs in self.obstacles.items():
            obs_pos = [obs.pos_x, obs.pos_y]
            _obs_distance, _obs_angle = CustomEnv.calculate_relative_distance_and_angle(self.leader_agent.pos, obs_pos)

            
            # obs_distance_angle.extend([obs_distance, obs_angle])
            if _obs_distance <= self.obs_delta:
                vx = self.leader_agent.vel * cos(self.leader_agent.orientation)
                vy = self.leader_agent.vel * sin(self.leader_agent.orientation)
                robot_state = [self.leader_agent.pos[0], self.leader_agent.pos[1], vx, vy, self.agent_radius]
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
                px = (obs.pos_x - self.leader_agent.pos[0]) / self.obs_delta
                py = (obs.pos_y - self.leader_agent.pos[1])/ self.obs_delta
                vx = 0
                vy = 0
                _obs_distance_ = _obs_distance / (self.obs_delta * 1.415)
            
            else:
                vo_flag = False
                px = 0
                py = 0
                _obs_distance_ = 0
            
            obs_dis_angle = (_obs_angle - self.leader_agent.orientation)
            obs_pos_vel.extend([px, py, _obs_distance_, obs_dis_angle / (2*np.pi), vo_flag])
            # obs_pos_vel.extend([px, py, vo_flag])
        
        leader_observation = np.array(
            side_pos + 
            target_pos_ +
            obs_pos_vel +
            self_pos_
        )

        # 跟随者的观测更新
        for i in range (self.follower_uav_num):
            _dis_, _angle_ = CustomEnv.calculate_relative_distance_and_angle(self.follower_uavs[f"follower_{i}"].pos, self.leader_agent.pos + self.formation_pos[i])


            side_pos_2 = [self.follower_uavs[f"follower_{i}"].pos[0] / self.width, (self.width - self.follower_uavs[f"follower_{i}"].pos[0]) / self.width, self.follower_uavs[f"follower_{i}"].pos[1] / self.height, (self.height - self.follower_uavs[f"follower_{i}"].pos[1]) / self.height]
            # target_pos_2 = [(self.leader_agent.pos[0] + self.formation_pos[i][0] - self.follower_uavs[f"follower_{i}"].pos[0]) / self.width, (self.leader_agent.pos[1] + self.formation_pos[i][1] - self.follower_uavs[f"follower_{i}"].pos[1]) / self.height, _dis_ / (self.width * 1.414), _angle_ ]
            target_pos_2 = [(self.leader_agent.pos[0] + self.formation_pos[i][0] - self.follower_uavs[f"follower_{i}"].pos[0]) , (self.leader_agent.pos[1] + self.formation_pos[i][1] - self.follower_uavs[f"follower_{i}"].pos[1]) , _dis_ , _angle_ ]


            obs_pos_vel_2 = []

            for obs_id, obs in self.obstacles.items():
                obs_pos = [obs.pos_x, obs.pos_y]
                _obs_distance, _obs_angle = CustomEnv.calculate_relative_distance_and_angle(self.follower_uavs[f"follower_{i}"].pos, obs_pos)

                
                # obs_distance_angle.extend([obs_distance, obs_angle])
                if _obs_distance <= self.obs_delta:
                    vx = self.follower_uavs[f"follower_{i}"].vel[0]
                    vy = self.follower_uavs[f"follower_{i}"].vel[1]
                    robot_state = [self.follower_uavs[f"follower_{i}"].pos[0], self.follower_uavs[f"follower_{i}"].pos[1], vx, vy, self.agent_radius]
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
                    px = (obs.pos_x - self.follower_uavs[f"follower_{i}"].pos[0]) / self.obs_delta
                    py = (obs.pos_y - self.follower_uavs[f"follower_{i}"].pos[1])/ self.obs_delta
                    vx = 0
                    vy = 0
                    _obs_distance_ = _obs_distance / (self.obs_delta * 1.415)
                
                else:
                    vo_flag = False
                    px = 0
                    py = 0
                    _obs_distance_ = 0
                
                obs_dis_angle = (_obs_angle)
                obs_pos_vel.extend([px, py, _obs_distance_, obs_dis_angle / (2*np.pi), vo_flag])
            
            self_pos_2 = [self.follower_uavs[f"follower_{i}"].vel[0], self.follower_uavs[f"follower_{i}"].vel[1]]#这里少了一维

            self.follower_uavs[f"follower_{i}"].observation = np.array([
                side_pos_2 + 
                target_pos_2 +
                obs_pos_vel_2 +
                self_pos_2
            ])
           

        return leader_observation
    
    def step(self, leader_action, leader_actions):
        """
        所有智能体前进一步，并更新观测 

        """
        self._apply_leader_action(self.leader_agent, leader_action, self.leader_target_pos)
        self._apply_follower_actions(leader_actions)

        observation = self.observe()

        # reward = self._calculate_leader_reward()
        



        return






    def _apply_follower_actions(self, follower_actions):
        for i in range (self.follower_uav_num):
            self.follower_uavs[f"follower_{i}"].vel[0] = follower_actions[i][0]
            self.follower_uavs[f"follower_{i}"].vel[1] = follower_actions[i][1]

            dx = self.follower_uavs[f"follower_{i}"].vel[0] * self.display_time
            dy = self.follower_uavs[f"follower_{i}"].vel[1] * self.display_time

            self.follower_uavs[f"follower_{i}"].pos[0] += dx
            self.follower_uavs[f"follower_{i}"].pos[1] += dy

            target_dis = sqrt((self.follower_uavs[f"follower_{i}"].pos[0] - (self.leader_agent.pos[0] + self.formation_pos[i][0]))**2 + (self.follower_uavs[f"follower_{i}"].pos[1] - (self.leader_agent.pos[1] + self.formation_pos[i][1]))**2)
            if target_dis < 1e-1:
                self.follower_uavs[f"follower_{i}"].target = True

            if  self._check_obs_collision(self.follower_uavs[f"follower_{i}"], self.follower_uavs[f"follower_{i}"].pos) or self._check_uav_collision(self.follower_uavs[f"follower_{i}"]
                                                                                                                                                     , self.follower_uavs[f"follower_{i}"].pos):
                flag = True
            else:
                flag = False

            if flag or self.follower_uavs[f"follower_{i}"].target:
                self.follower_uavs[f"follower_{i}"].done = True




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
    
    def _check_uav_collision(self, current_agent, pos):
        for uav in self.follower_uavs.values():
            uav_pos = uav.pos
            if np.linalg.norm(np.array(pos) - np.array(uav_pos)) != 0:
                if np.linalg.norm(np.array(pos) - np.array(uav_pos)) <= self.agent_radius * 2 + self.safe_theta/40:
                    return True
        return False
                    

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

    

