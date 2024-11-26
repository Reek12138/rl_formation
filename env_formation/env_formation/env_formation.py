# -*- coding: utf-8 -*-
from .rvo_inter import rvo_inter
import numpy as np
from .obstacle import obstacle
from .circle_agent_sac import circle_agent
from .follower_uav import follower_uav
import matplotlib.pyplot as plt
from math import sin, cos, tan, pi, sqrt, log
import matplotlib.patches as patches
import torch
from env_formation.masac import MASAC

np.set_printoptions(precision=5, suppress=True)
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
        self.last_leader_action = [0, 0]
        self.leader_agent_trajectory = []

        # 生成circle_agent实例列表
        self.leader_agent = circle_agent(self, pos=[25, 25], vel=[0,0], orientation=0, memo_size=memo_size,
                                        #  state_dim=13 + self.num_obstacles * 5,
                                         state_dim=13 + 6 * 5,
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
            [0, 2*3],
            [-1.732*3, -1*3],
            [1.732*3, -1*3]
        ]

        for i in range (follower_uav_num):
            self.follower_uavs[f"follower_{i}"] = follower_uav(radius=self.agent_radius,
                                                               pos = [self.leader_agent.pos[0] + self.formation_pos[i][0], self.leader_agent.pos[1] + self.formation_pos[i][1]],
                                                               vel=[0,0],
                                                               memo_size=100000, state_dim=40, action_dim=2, alpha=1e-4, beta=1e-4,
                                                               alpha_lr=1e-4, hidden_dim=512, gamma=0.99, tau=0.01, batch_size=512,target_entropy=-log(2) )
        # print(self.leader_agent.pos)
        # for i in range (follower_uav_num):
        #     print(self.follower_uavs[f"follower_{i}"].pos)
        self.MASAC = MASAC(state_dim = (4+2+3*6+2+5*(self.follower_uav_num-1)),
                        #    state_dim = (4+4+5*self.num_obstacles+2+5*(self.follower_uav_num-1))
                                                            hidden_dim = 512,
                                                            action_dim=2,
                                                            actor_lr=1e-4,
                                                            critic_lr=1e-4,
                                                            alpha_lr=1e-4,
                                                            target_entropy= -log(2*self.follower_uav_num),
                                                            tau=0.01,
                                                            gamma=0.99,
                                                            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                                                            agent_num=self.follower_uav_num)
        # print("PolicyNetwork input size:", self.MASAC.actor.fc1.in_features)
        # print("QvalueNet input size (critic_1):", self.MASAC.critic_1.fc1.in_features)

        
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
        for i in range(self.follower_uav_num):
            for obs in self.obstacles.values():
                dis = np.linalg.norm(obs.position() - self.follower_uavs[f"follower_{i}"].position())
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
                if np.linalg.norm(obs.position() - np.array(self.follower_uavs[f"follower_{i}"].position())) < self.agent_radius +self.obs_radius + self.safe_theta/4:
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
        flag0 = self._check_fix_obs_agent(self.leader_agent.pos)
        while flag0:
            self.leader_agent.set_position(self.width*0.10+np.random.rand()*self.width*0.8, self.height*0.10+np.random.rand()*self.height*0.8)
            for i in range(self.follower_uav_num):
                self.follower_uavs[f"follower_{i}"].set_position(self.leader_agent.pos[0] + self.formation_pos[i][0], self.leader_agent.pos[1] + self.formation_pos[i][1])
            flag0 = self._check_fix_obs_agent(self.leader_agent.pos)

        
        
        # 确保目标不与固定障碍物碰撞
        flag2 = self._check_obs_target(self.leader_target_pos)
        while flag2:
            self.leader_target_pos = [self.width*0.1+np.random.rand()*self.width*0.8, self.height*0.1+np.random.rand()*self.height*0.8]
            flag2 = self._check_obs_target(self.leader_target_pos)

        self._check_obs()
        
        flag1 = self._check_obs_agent(self.leader_agent)
        while  np.linalg.norm(np.array(self.leader_target_pos) - np.array(self.leader_agent.pos)) < self.agent_radius + self.target_radius + self.safe_theta*3 or flag1:
            self.leader_agent.set_position(self.width*0.10+np.random.rand()*self.width*0.8, self.height*0.10+np.random.rand()*self.height*0.8)
            for i in range(self.follower_uav_num):
                self.follower_uavs[f"follower_{i}"].set_position(self.leader_agent.pos[0] + self.formation_pos[i][0], self.leader_agent.pos[1] + self.formation_pos[i][1])
            flag1 = self._check_obs_agent(self.leader_agent)

        target_distance =  np.linalg.norm(np.array(self.leader_target_pos) - np.array(self.leader_agent.pos))

        

        self.leader_agent.set_vel(0)
        self.leader_agent.orientation = np.random.rand()*2*np.pi

        for i in range(self.follower_uav_num):
            self.follower_uavs[f"follower_{i}"].set_vel(0, 0)
            self.follower_uavs[f"follower_{i}"].target = False
            self.follower_uavs[f"follower_{i}"].done = False
            self.follower_uavs[f"follower_{i}"].obs_done = False
            self.follower_uavs[f"follower_{i}"].side_done = False
            self.follower_uavs[f"follower_{i}"].uav_done = False

        self.leader_agent.done = False
        self.leader_agent.target = False
        observations = self.observe()
        target_info = self.leader_agent.target

        return observations, self.leader_agent.done
    
    def observe(self):
        """
        输出领航者的观测， 更新跟随者的观测
        跟随者的观测和actor网络都写在自己的节点内
        领航者的观测则作为返回值，跟之前的训练模式一样

        """
        # 领航者的观测更新
        _dis, _angle = CustomEnv.calculate_relative_distance_and_angle(self.leader_agent.pos, self.leader_target_pos)
        side_pos = [self.leader_agent.pos[0] / self.width, (self.width - self.leader_agent.pos[0]) / self.width, self.leader_agent.pos[1] / self.height, (self.height - self.leader_agent.pos[1]) / self.height]
        target_pos_ = [(self.leader_target_pos[0] - self.leader_agent.pos[0]) / self.width, (self.leader_target_pos[1] - self.leader_agent.pos[1]) / self.height, _dis / (self.width * 1.414), (_angle - self.leader_agent.orientation) / (2 * np.pi)]
        self_pos_ = [self.leader_agent.orientation / (2*np.pi), self.leader_agent.vel * cos(self.leader_agent.orientation)/2, self.leader_agent.vel * sin(self.leader_agent.orientation)/2]
        last_action = list(self.last_leader_action[:2])  # 转换为列表

        obs_distance_angle = []
        obs_pos_vel = []
        obs_num = 0
        for obs_id, obs in self.obstacles.items():
            obs_pos = [obs.pos_x, obs.pos_y]
            _obs_distance, _obs_angle = CustomEnv.calculate_relative_distance_and_angle(self.leader_agent.pos, obs_pos)

            
            # obs_distance_angle.extend([obs_distance, obs_angle])
            if _obs_distance <= self.obs_delta:
                obs_num += 1
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
            
            # else:
            #     vo_flag = False
            #     px = 0
            #     py = 0
            #     _obs_distance_ = 0
            
                obs_dis_angle = (_obs_angle - self.leader_agent.orientation)
                obs_pos_vel.extend([px, py, _obs_distance_, obs_dis_angle / (2*np.pi), vo_flag])
                # obs_pos_vel.extend([px, py, vo_flag])
        for _ in range (6 - obs_num):
            obs_pos_vel.extend([0, 0, 0, 0, False])
        
        leader_observation = np.array(
            side_pos + 
            target_pos_ +
            obs_pos_vel +
            self_pos_ +
            last_action
        )

        # 跟随者的观测更新
        for i in range (self.follower_uav_num):
            _dis_, _angle_ = CustomEnv.calculate_relative_distance_and_angle(self.follower_uavs[f"follower_{i}"].pos,\
                                                                              [self.leader_agent.pos[0] + self.formation_pos[i][0], self.leader_agent.pos[1] + self.formation_pos[i][1]])


            side_pos_2 = [self.follower_uavs[f"follower_{i}"].pos[0] / self.width, (self.width - self.follower_uavs[f"follower_{i}"].pos[0]) / self.width, self.follower_uavs[f"follower_{i}"].pos[1] / self.height, (self.height - self.follower_uavs[f"follower_{i}"].pos[1]) / self.height]
            # target_pos_2 = [(self.leader_agent.pos[0] + self.formation_pos[i][0] - self.follower_uavs[f"follower_{i}"].pos[0]) / self.width, (self.leader_agent.pos[1] + self.formation_pos[i][1] - self.follower_uavs[f"follower_{i}"].pos[1]) / self.height, _dis_ / (self.width * 1.414), _angle_ ]
            target_pos_2 = [(self.leader_agent.pos[0] + self.formation_pos[i][0] - self.follower_uavs[f"follower_{i}"].pos[0]) , (self.leader_agent.pos[1] + self.formation_pos[i][1] - self.follower_uavs[f"follower_{i}"].pos[1]) ]

            # 障碍物的状态观测
            obs_pos_vel_2 = []
            obs_num_follower = 0
            for obs_id, obs in self.obstacles.items():
                obs_pos = [obs.pos_x, obs.pos_y]
                _obs_distance, _obs_angle = CustomEnv.calculate_relative_distance_and_angle(self.follower_uavs[f"follower_{i}"].pos, obs_pos)
                # obs_distance_angle.extend([obs_distance, obs_angle])
                if _obs_distance <= self.obs_delta:
                    obs_num_follower += 1
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
                
                # else:
                #     vo_flag = False
                #     px = 0
                #     py = 0
                #     _obs_distance_ = 0
                
                    obs_dis_angle = (_obs_angle)
                    # obs_pos_vel_2.extend([px, py, _obs_distance_, obs_dis_angle / (2*np.pi), vo_flag])
                    obs_pos_vel_2.extend([px, py, vo_flag])
            
            for _ in range (6 - obs_num_follower):
                obs_pos_vel_2.extend([0, 0, False])
                
            # 自身速度观测
            self_pos_2 = [self.follower_uavs[f"follower_{i}"].vel[0], self.follower_uavs[f"follower_{i}"].vel[1]]#这里少了一维


            # 对于其他跟随者无人机的观测
            follower_pos_ = []
            for j in range (self.follower_uav_num):
                if j != i:
                    follower_dis, follower_angle = CustomEnv.calculate_relative_distance_and_angle(self.follower_uavs[f"follower_{i}"].pos, \
                                                                                                   self.follower_uavs[f"follower_{i}"].pos)
                    robot_state = [self.follower_uavs[f"follower_{i}"].pos[0], self.follower_uavs[f"follower_{i}"].pos[1], \
                                   self.follower_uavs[f"follower_{i}"].vel[0], self.follower_uavs[f"follower_{i}"].vel[1],
                                   self.agent_radius]
                    nei_state_list =[ [self.follower_uavs[f"follower_{j}"].pos[0], self.follower_uavs[f"follower_{j}"].pos[1], \
                                      self.follower_uavs[f"follower_{j}"].vel[0], self.follower_uavs[f"follower_{j}"].vel[1],\
                                        self.agent_radius]]
                    obs_cir_list = []
                    obs_line_list = []
                    action = [self.follower_uavs[f"follower_{i}"].vel[0], self.follower_uavs[f"follower_{i}"].vel[1]]
                    follow_vo_flag, min_exp_time, min_dis = self.rvo_inter.config_vo_reward(robot_state=robot_state,
                                                                                                                                                                            nei_state_list=nei_state_list,
                                                                                                                                                                            obs_cir_list=obs_cir_list,
                                                                                                                                                                            obs_line_list=obs_line_list,
                                                                                                                                                                            action=action)
                    px = (self.follower_uavs[f"follower_{j}"].pos[0] - self.follower_uavs[f"follower_{i}"].pos[0]) / (self.formation_pos[0][1] * 2)
                    py = (self.follower_uavs[f"follower_{j}"].pos[1] - self.follower_uavs[f"follower_{i}"].pos[1]) / (self.formation_pos[0][1] * 2)
                    vx = (self.follower_uavs[f"follower_{j}"].vel[0] - self.follower_uavs[f"follower_{i}"].vel[0] )
                    vy = (self.follower_uavs[f"follower_{j}"].vel[1] - self.follower_uavs[f"follower_{i}"].vel[1])

                    follower_j = [px, py, vx, vy, follow_vo_flag]
                    follower_pos_.extend(follower_j)
                


            self.follower_uavs[f"follower_{i}"].observation = np.array([
                side_pos_2 + 
                target_pos_2 +
                obs_pos_vel_2 +
                self_pos_2+
                follower_pos_
            ])
           

        return leader_observation
    
    def step(self, leader_action, follower_actions, last_distance, last_obs_distance, last_follower_obs_distance):
        """
        所有智能体前进一步，并更新观测 

        """
        self._apply_leader_action( leader_action)
        self._apply_follower_actions(follower_actions)

        leader_observation = self.observe()
        follower_observations = []
        follower_done = False
        for i in range (self.follower_uav_num):
            next_state = self.follower_uavs[f"follower_{i}"].observation
            follower_observations.extend(next_state)
            if self.follower_uavs[f"follower_{i}"].done == True:
                follower_done = True

        leader_reward = self._calculate_leader_reward(leader_action, last_distance=last_distance, last_obs_distance=last_obs_distance)

        follower_reward = self._calculate_follower_reward(follower_actions, last_obs_distances=last_follower_obs_distance)
        # reward = self._calculate_leader_reward()

        return  leader_observation, leader_reward, self.leader_agent.done, self.leader_agent.target,\
                follower_observations, follower_reward, follower_done

    def _apply_follower_actions(self, follower_actions):
        for i in range (self.follower_uav_num):
            self.follower_uavs[f"follower_{i}"].vel[0] = follower_actions[2*i] *2
            self.follower_uavs[f"follower_{i}"].vel[1] = follower_actions[2*i + 1] *2

            dx = self.follower_uavs[f"follower_{i}"].vel[0] * self.display_time
            dy = self.follower_uavs[f"follower_{i}"].vel[1] * self.display_time

            self.follower_uavs[f"follower_{i}"].pos[0] += dx
            self.follower_uavs[f"follower_{i}"].pos[1] += dy

            target_dis = sqrt((self.follower_uavs[f"follower_{i}"].pos[0] - (self.leader_agent.pos[0] + self.formation_pos[i][0]))**2 + (self.follower_uavs[f"follower_{i}"].pos[1] - (self.leader_agent.pos[1] + self.formation_pos[i][1]))**2)
            if target_dis < 1e-1:
                self.follower_uavs[f"follower_{i}"].target = True
                
            flag = False
            if  self._check_obs_collision(self.follower_uavs[f"follower_{i}"], self.follower_uavs[f"follower_{i}"].pos) == True:
                self.follower_uavs[f"follower_{i}"].obs_done = True
                flag = True

            if self._check_uav_collision(self.follower_uavs[f"follower_{i}"], self.follower_uavs[f"follower_{i}"].pos, id=i) == True:
                self.follower_uavs[f"follower_{i}"].uav_done =True
                flag = True

            if self.follower_uavs[f"follower_{i}"].pos[0] < self.agent_radius or self.follower_uavs[f"follower_{i}"].pos[1] < self.agent_radius \
                or self.follower_uavs[f"follower_{i}"].pos[0] > self.width - self.agent_radius or self.follower_uavs[f"follower_{i}"].pos[1] > self.height - self.agent_radius:
                self.follower_uavs[f"follower_{i}"].side_done = True
                flag = True

            if flag :
                self.follower_uavs[f"follower_{i}"].done = True

    def _apply_leader_action(self, action):
        """假设输入的动作是[线速度 m/s, 转向角 弧度]"""
        linear_vel = action[0] + 1  # 线速度，带偏置
        steer_angle = action[1]  # 转向角

        # 单轨模型的参数
        L = 2.5  # 前后轮间距，车辆的轴距
        dt = self.display_time  # 时间步长

        # 更新航向角（yaw angle）
        new_orientation = self.leader_agent.orientation + (linear_vel / L) * tan(steer_angle) * dt
        new_orientation = new_orientation % (2 * pi)  # 规范化到 [0, 2pi]

        # 更新位置 (x, y)
        dx = linear_vel * dt * cos(new_orientation)
        dy = linear_vel * dt * sin(new_orientation)
        x = self.leader_agent.pos[0] + dx
        y = self.leader_agent.pos[1] + dy
        new_pos = [x, y]

        # 计算与目标的距离
        target_dis = sqrt((x - self.leader_target_pos[0])**2 + (y - self.leader_target_pos[1])**2)
        if target_dis < self.target_radius:  # 到达目标点
            self.leader_agent.target = True

        # 边界检测
        if x < self.agent_radius or x > self.width - self.agent_radius or \
            y < self.agent_radius or y > self.height - self.agent_radius:
            flag = True
        else:
            flag = False

        # 检测是否发生碰撞，并更新状态
        if not self._check_obs_collision(self.leader_agent, new_pos) and not flag and not self.leader_agent.target:
            self.leader_agent.set_position(x, y)  # 更新位置
            self.leader_agent.orientation = new_orientation  # 更新航向角
            self.leader_agent.vel = linear_vel  # 更新速度
        else:
            self.leader_agent.done = True  # 标记为完成

# ======计算领航者的奖励====================
    def _calculate_leader_reward(self, leader_action, last_distance, last_obs_distance):
        reward ,reward1, reward2,reward3, reward4, reward5 = 0,0,0,0,0,0
        vo_flag, reward1, min_dis = self._caculate_leader_vo_reward()
        reward2 = self._caculate_target_reward(last_distance ,vo_flag)
        reward3 = self._caculate_obstacle_reward( last_obs_distance)
        reward4 = self._caculate_velocity_reward(leader_action, vo_flag, self.last_leader_action)
        reward5 = self._caculate_side_reward()
        reward6 = self._leader_formation_reward()

        reward = round(reward2 + reward3 + reward4 + reward5 + reward6, 5)
        self.last_leader_action = leader_action
        return reward
        
    def _caculate_leader_vo_reward(self):
            vx = self.leader_agent.vel * cos(self.leader_agent.orientation)
            vy = self.leader_agent.vel * sin(self.leader_agent.orientation)
            robot_state = [self.leader_agent.pos[0], self.leader_agent.pos[1], vx, vy, self.agent_radius]
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

    def _caculate_target_reward(self, last_distance, vo_flag):
            """和编队目标之间的距离"""
            reward = 0
            dis, angle = CustomEnv.calculate_relative_distance_and_angle(self.leader_agent.pos, self.leader_target_pos)

            # 设置最小阈值
            min_dis_threshold = self.target_radius/2
            if dis < min_dis_threshold:
                dis = min_dis_threshold

            
            if self.leader_agent.done and self.leader_agent.target:
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

    def _caculate_obstacle_reward(self, last_obs_distance):
        if self.leader_agent.done and not self.leader_agent.target:
                    return -500
        
        reward = 0
        
        for obs_id, obs in self.obstacles.items():
            dis, angle = CustomEnv.calculate_relative_distance_and_angle(self.leader_agent.pos, [obs.pos_x, obs.pos_y])
            if dis <= self.obs_delta and abs(angle - self.leader_agent.orientation)<=(np.pi/2)*1.2:
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
                if vo_flag:
                    delta = 800
                    x = 60
                else:
                    delta = 100
                    x = 0
                d_dis = dis - last_obs_distance[obs_id]
                # reward += d_dis * delta+ -(1/dis) * x
                reward += round(d_dis * delta - (1 / dis) * x, 5)

                
        return  reward 
    
    def _caculate_velocity_reward(self, action, vo_flag, last_action):
        if vo_flag:
            return round((action[0] + 1) *5 - abs(action[1]) *5  - abs(action[1] - last_action[1]) *15, 5)
        else:
            return round(((action[0] + 1) ) *5 - abs(action[1]) *5 - abs(action[1] - last_action[1])*15, 5)
        
    def _caculate_side_reward(self):
        reward = 0
        distances = [self.leader_agent.pos[0], self.width - self.leader_agent.pos[0], self.leader_agent.pos[1], self.height - self.leader_agent.pos[1]]
        
        for dis in distances:
            if dis > self.width * 0.18:
                re = 0
            
            elif self.width *0.05 < dis <= self.width* 0.18:
                re = round(-self.width *0.05 /dis, 5)
            elif dis <= self.width *0.05:
                if self.leader_agent.done and not self.leader_agent.target:
                    return -100
                re = round(-20 *((self.width *0.05)/(self.width *0.05 + dis)), 5)

            reward += re

        return reward *10  
    
    def _leader_formation_reward(self):
        reward = 0
        for i in range (self.follower_uav_num):
            pos = [p + f for p, f in zip(self.leader_agent.pos, self.formation_pos[i])]
            for obs_id, obs in self.obstacles.items():
                dis = np.linalg.norm(np.array(pos) - np.array([obs.pos_x, obs.pos_y]))
                if dis <= 3:
                    reward += -10
        return reward
# ======计算跟随者的奖励===================
    def _calculate_follower_reward(self,follower_actions, last_obs_distances):
        follower_rewards = []
        for i in range (self.follower_uav_num):
            last_obs_distance = last_obs_distances[i]
            obs_collision_reward = self._follower_obs_collision_reward(i, last_obs_distance)
            side_reward = self._follower_side_reward(i)
            uav_collision_reward = self._uav_collision_reward(i)
            target_reward = self._follower_target_reward(i)
            vel_reward = self._follower_vel_reward(i)
            r = round(target_reward + obs_collision_reward + side_reward + uav_collision_reward + vel_reward, 5)
            # print("uav_",i ," target_reward : ", target_reward,  " side_reward : ", side_reward, " obs_reward : ", \
            #       obs_collision_reward, " uav_coll_reward : ", uav_collision_reward, " vel_reward : ", vel_reward)
            # follower_rewards.extend(r)
            follower_rewards.append(r)
        return follower_rewards

    def _follower_target_reward(self, uav_id):
        if self.follower_uavs[f"follower_{uav_id}"].target == True:
            self.follower_uavs[f"follower_{uav_id}"].target = False
            return 50
        else :
            return 0
        # dis, angle = CustomEnv.calculate_relative_distance_and_angle(self.follower_uavs[f"follower_{uav_id}"].pos,
        #                                                              [self.leader_agent.pos[0] + self.formation_pos[uav_id][0], self.leader_agent.pos[1] + self.formation_pos[uav_id][1]])
        # return -dis *200
    
    def _follower_obs_collision_reward(self, uav_id, last_obs_distance):
        if self.follower_uavs[f"follower_{uav_id}"].done and not self.follower_uavs[f"follower_{uav_id}"].target:
            return -500
        
        reward = 0
        for obs_id, obs in self.obstacles.items():
            dis, angle = CustomEnv.calculate_relative_distance_and_angle(self.follower_uavs[f"follower_{uav_id}"].pos,
                                                                          [obs.pos_x, obs.pos_y])
            if dis <= self.obs_delta:
                reward_ = 0
                robot_state = [self.follower_uavs[f"follower_{uav_id}"].pos[0], self.follower_uavs[f"follower_{uav_id}"].pos[1], 
                               self.follower_uavs[f"follower_{uav_id}"].vel[0], self.follower_uavs[f"follower_{uav_id}"].vel[1],
                                 self.agent_radius]
                nei_state_list = []
                obs_cir_list = [ [obs.pos_x, obs.pos_y, obs.xy_vel[0], obs.xy_vel[1], self.obs_radius * 1.2]]#放大
                obs_line_list = []
                action = [self.follower_uavs[f"follower_{uav_id}"].vel[0], self.follower_uavs[f"follower_{uav_id}"].vel[1]]
                vo_flag, min_exp_time, min_dis = self.rvo_inter.config_vo_reward(robot_state=robot_state,
                                                                                                                                                        nei_state_list=nei_state_list,
                                                                                                                                                        obs_cir_list=obs_cir_list,
                                                                                                                                                        obs_line_list=obs_line_list,
                                                                                                                                                        action=action)
                # if vo_flag:
                #     delta = 800
                #     x = 120
                #     # print(f"uav_{uav_id}_vo_flag")
                # else:
                #     delta = 100
                #     x = 5
                # dis_ = dis - last_obs_distance[obs_id]
                # reward +=  round(-(1/dis) * x + dis_ * delta, 5)

                vx = self.follower_uavs[f"follower_{uav_id}"].vel[0]
                vy = self.follower_uavs[f"follower_{uav_id}"].vel[1]
                delta_x = obs.pos_x - self.follower_uavs[f"follower_{uav_id}"].pos[0]
                delta_y = obs.pos_y - self.follower_uavs[f"follower_{uav_id}"].pos[1]

                if vx * delta_x >= 0:
                    reward_ += -vx /(delta_x + 1e-1)
                else:
                    reward_ += 0

                if vy * delta_y >= 0:
                    reward_ += -vy /(delta_y + 1e-1)
                else:
                    reward_ += 0
                
                if vo_flag:
                    reward += reward_ * 500
                else:
                    reward += reward_ *50
                
        
        return reward *2

    def _uav_collision_reward(self, uav_id):
        reward = 0
        for j in range (self.follower_uav_num):
            if j != uav_id:
                dis, angle = CustomEnv.calculate_relative_distance_and_angle(self.follower_uavs[f"follower_{uav_id}"].pos,
                                                                          self.follower_uavs[f"follower_{j}"].pos)
                # if dis <= self.obs_delta/2:
                robot_state = [self.follower_uavs[f"follower_{uav_id}"].pos[0], self.follower_uavs[f"follower_{uav_id}"].pos[1], 
                            self.follower_uavs[f"follower_{uav_id}"].vel[0], self.follower_uavs[f"follower_{uav_id}"].vel[1],
                                self.agent_radius]
                nei_state_list = [[self.follower_uavs[f"follower_{j}"].pos[0], self.follower_uavs[f"follower_{j}"].pos[1], 
                            self.follower_uavs[f"follower_{j}"].vel[0], self.follower_uavs[f"follower_{j}"].vel[1],
                                self.agent_radius]]
                obs_cir_list =[]
                obs_line_list = []
                action = [self.follower_uavs[f"follower_{uav_id}"].vel[0], self.follower_uavs[f"follower_{uav_id}"].vel[1]]

                vo_flag, min_exp_time, min_dis = self.rvo_inter.config_vo_reward(robot_state=robot_state,
                                                                                                                                                    nei_state_list=nei_state_list,
                                                                                                                                                    obs_cir_list=obs_cir_list,
                                                                                                                                                    obs_line_list=obs_line_list,
                                                                                                                                                    action=action)
                if dis < self.formation_pos[0][1] *1.4:
                    # print("formation vo flag")
                    if vo_flag == True:
                        reward += round((-6000/dis), 5)
                    else:
                        reward += round((-2000/dis), 5)
        return reward

    def _follower_side_reward(self, uav_id):
        reward = 0
        distances = [self.follower_uavs[f"follower_{uav_id}"].pos[0], \
                                self.width - self.follower_uavs[f"follower_{uav_id}"].pos[0], \
                                self.follower_uavs[f"follower_{uav_id}"].pos[1], \
                                self.height - self.follower_uavs[f"follower_{uav_id}"].pos[1]]
        for dis in distances:
            if dis > self.width * 0.18:
                re = 0
            
            elif self.width *0.05 < dis <= self.width* 0.18:
                re = round(-self.width *0.05 /dis, 5)
            elif dis <= self.width *0.05:
                if self.leader_agent.done and not self.leader_agent.target:
                    return -100
                re = round(-20 *((self.width *0.05)/(self.width *0.05 + dis)), 5)

            reward += re
        return reward *20

    def _follower_vel_reward(self, uav_id):
        reward = 0
        vx = self.follower_uavs[f"follower_{uav_id}"].vel[0]
        vy = self.follower_uavs[f"follower_{uav_id}"].vel[1]
        delta_x = self.leader_agent.pos[0]+self.formation_pos[uav_id][0] - self.follower_uavs[f"follower_{uav_id}"].pos[0]
        delta_y = self.leader_agent.pos[1]+self.formation_pos[uav_id][1] - self.follower_uavs[f"follower_{uav_id}"].pos[1]
        if vx * delta_x >= 0:
            reward += vx /(delta_x + 1e-1)
        else:
            reward += -abs(vx * delta_x)*3

        if vy * delta_y >= 0:
            reward += vy /(delta_y + 1e-1)
        else:
            reward += -abs(vy * delta_y)*3

        return reward *3
        

    def _check_obs_collision(self,current_agent,new_pos):
        """检查智能体是否与障碍物碰撞"""
        for obs in self.obstacles.values():
            obs_pos = [obs.pos_x, obs.pos_y]
            if np.linalg.norm(np.array(new_pos) -np.array(obs_pos)) <= self.obs_radius + self.agent_radius:
                return True
        return False
    
    def _check_uav_collision(self, current_agent, pos, id):
        # for uav in self.follower_uavs.values():
        #     uav_pos = uav.pos
        #     if np.linalg.norm(np.array(pos) - np.array(uav_pos)) != 0:
        #         if np.linalg.norm(np.array(pos) - np.array(uav_pos)) <= self.agent_radius * 2 + self.safe_theta/40:
        #             return True
        for i in range (self.follower_uav_num):
            if i != id:
                if np.linalg.norm(self.follower_uavs[f"follower_{id}"].position() - self.follower_uavs[f"follower_{i}"].position()) <= self.agent_radius * 2 :
                    return True
        return False
                    
    def render(self, display_time = 0.1):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(10,10), dpi=100)
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)
            self.ax.set_aspect('equal')
        
        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        leader_agent = patches.Circle(self.leader_agent.pos, self.agent_radius, color='purple', fill=True)
        self.ax.add_patch(leader_agent)
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
        self.leader_agent_trajectory.append(self.leader_agent.pos.copy())
        # 绘制智能体的轨迹
        if len(self.leader_agent_trajectory) > 1:
            traj_x, traj_y = zip(*self.leader_agent_trajectory)
            self.ax.plot(traj_x, traj_y, color='blue', linestyle='--', marker='o', markersize=1, label='Trajectory')
        
        # 绘制跟随者无人机
        for i in range(self.follower_uav_num):
            uav = patches.Circle(self.follower_uavs[f"follower_{i}"].pos, self.agent_radius, color='orange', fill=True)
            self.ax.add_patch(uav)
        
        # 绘制障碍物
        obses = [patches.Circle([obs.pos_x, obs.pos_y], self.obs_radius, color='red', fill=True)for obs in self.obstacles.values()]
        for obs_circle in obses:
            self.ax.add_patch(obs_circle)
        # 绘制目标
        target = patches.Circle(self.leader_target_pos, self.target_radius, color='green', fill=True)
        self.ax.add_patch(target)

        plt.pause(self.display_time)  # 暂停以更新图形
        # plt.show()

    def render_close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None
        


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

    

