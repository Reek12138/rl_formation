# -*- coding: utf-8 -*-
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils import agent_selector
from gym.spaces import Box, Discrete
import numpy as np
from pettingzoo.mpe import simple_adversary_v3
from obstacle import obstacle
from circle_agent import circle_agent
import matplotlib as plt

# env = simple_adversary_v3.parallel_env(render_mode = "human",N= 2, max_cycles = NUM_STEP, continuous_actions = True)


class CustomEnv(ParallelEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents=3, width=200, height=200, num_obstacles=5, agent_radius=1, safe_theta = 2,
                 target_pos = np.array([175, 175])):
        super().__init__()
        self.num_agents = num_agents
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.obs_radius = 10
        self.agent_radius = agent_radius
        
        # 生成circle_agent实例列表
        self.leader_agent = circle_agent(self.agent_radius, pos_x=5, pos_y=5)
        self.follower_agents = {
            f"agent_{i}": circle_agent(
                agent_radius,
                pos_x=self.leader_agent.pos_x + i*3 +3,
                pos_y=self.leader_agent.pos_y + i*3 +3
                ) 
                for i in range(self.num_agents)}

        self.obstacles = {
            f"obstacle_{i}":obstacle(radius=self.obs_radius,
                                     pos_x=np.random.rand() * self.width*0.8,
                                     pos_y=np.random.rand() * self.height*0.8,
                                     safe_theta= safe_theta
                                     )for i in range(self.num_obstacles)
        }
        self._check_obs_collision()
        self.leader_target_pos = target_pos#目标位置
        self.reset()

    
    def _check_obs_collision(self):
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


    def reset(self):#TODO
        #    重置位置
        self.leader_agent.set_position(5,5)
        for i,agent in enumerate(self.follower_agents.values()) :
            agent.set_position(self.leader_agent.pos_x + i*3 +3, self.leader_agent.pos_y + i*3 +3)

        # 重置observations TODO 
        self.observations = {}
        self.leader_agent.observation = {}#自己的位置、到目标点的距离、到目标点的角度、和所有障碍物的距离、和所有障碍物的角度
        self.observations['leader_agent'] = self.leader_agent.observation
        for agent_id, agent in self.follower_agents.items():
            agent.observation = {}  
            self.observations[agent_id] = agent.observation

        # 重置rewards
        self.rewards = {}
        self.leader_agent.reward = 0
        self.rewards['leader_agent'] = self.leader_agent.reward
        for agent_id, agent in self.follower_agents.items():
            agent.reward = 0  
            self.rewards[agent_id] = agent.reward 

        # 重置dones
        self.dones = {}
        self.leader_agent.done = False
        self.dones['leader_agent'] = self.leader_agent.done
        for agent_id, agent in self.follower_agents.items():
            agent.done = False
            self.dones[agent_id] = agent.done

        # 重置infos
        self.infos = {}
        self.leader_agent.info = {}
        self.infos['leader_agent'] = self.leader_agent.info
        for agent_id, agent in self.follower_agents.items():
            agent.info = {}
            self.infos[agent_id] = agent.info

    def step(self, actions):#TODO
        # agent = self._agent_selector.next()
        self._apply_action( actions)
        
        observations = {}
        rewards = {}
        dones = {}
        infos = {}

        for agent in self.follower_agents:
            observations[agent] = self.observe(agent, self.target_pos[agent])
            rewards[agent] = self._calculate_reward(agent)
            dones[agent] = self._is_done(agent)
            infos[agent] = {}

        return observations, rewards, dones, infos

    def observe(self, agent, target_pos):#TODO
        # agent自身位置,与目标的距离和角度,与最近障碍物的距离和角度,与集群中其他智能体的距离和角度
        self_pos = self.follower_agents[agent]
        target_dis, target_angle = self.calculate_relative_distance_and_angle(self_pos, target_pos)

        other_agents_distance_angle = []
        for other_agent, other_pos in self.agent_pos.items():
            if other_agent != agent:
                distance, angle = self.calculate_relative_distance_and_angle(self_pos, other_pos)
                other_agents_distance_angle.extend([distance, angle])
        
        closest_obs_pos = self._find_closest_obstacle(self_pos)
        obs_distance, obs_angle = self.calculate_relative_distance_and_angle(self_pos, closest_obs_pos)

        observation = np.array([target_dis, target_angle, obs_distance, obs_angle] + other_agents_distance_angle)

        return observation

    def _find_closest_obstacle(self, agent_pos):#TODO
        # 这个方法计算最近障碍物的位置
        closest_obstacle_pos = None
        min_dist = float('inf')
        for obstacle_pos in self.obstacles:
            dist = np.linalg.norm(np.array(obstacle_pos) - np.array(agent_pos))
            if dist < min_dist:
                min_dist = dist
                closest_obstacle_pos = obstacle_pos
        return closest_obstacle_pos
        

    def render(self, mode='human'):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)
            self.ax.set_aspect('equal')

        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)

        # 绘制智能体
        self.ax.plot(self.leader_agent.pos_x, self.leader_agent.pos_y, color='purple', markersize=self.agent_radius)
        for agent in self.follower_agents.values():
            self.ax.plot(agent.pos_x, agent.pos_y, color='blue', markersize=self.agent_radius)

        # 绘制障碍物
        for obs in self.obstacles.values():
            self.ax.plot(obs.pos_x, obs.pos_y, color='red', markersize=8)

        # 绘制目标
        for target_pos in self.target_pos.values():
            self.ax.plot(target_pos[0], target_pos[1], color='green', markersize=8)

        plt.pause(0.01)  # 暂停以更新图形

    def render_close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None

    def _apply_action(self, multi_action_pos):#TODO
        # Update the agent's position based on the action
        # action_pos是一个包含所有动作的字典

        new_positions = {}
        for agent, action_pos in multi_action_pos:
            new_pos = self.agent_pos[agent] + np.array(action_pos)
            new_positions[agent] = new_pos
        
        for agent, new_pos in new_positions.items():
            if not self._check_collision(agent, new_pos, new_positions):
                self.agent_pos[agent] = new_pos
            else:
                self.dones[agent] = True

    def _check_collision(self,current_agent,new_pos,new_positions):
        for obs_pos in self.obstacles:
            if np.linalg.norm(new_pos - obs_pos) < self.obs_radius:
                return True
        
        for agent, pos in new_positions.items():
            if agent != current_agent and np.linalg.norm(new_pos - pos) < self.agent_radius:
                return True
        return False
    # def _check_obs_collision(self, pos):
    #     for obs_pos in self.obstacles:
    #         if np.linalg.norm(pos - obs_pos) < self.obs_radius:
    #             return True
    #     return False

    # Check if the agent has reached the target
    def _is_done(self, agent):
        if self.dones[agent] == True:
            return True
        elif self.dones[agent] == False:
            distance2target = np.linalg.norm(self.agent_pos[agent] - self.target_pos[agent])
            return distance2target < self.agent_pos + 2

    def _calculate_reward(self, agent):
        # Calculate the reward for the agent
        pass



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