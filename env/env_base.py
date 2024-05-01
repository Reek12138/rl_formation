from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils import agent_selector
from gym.spaces import Box, Discrete
import numpy as np
from pettingzoo.mpe import simple_adversary_v3

# env = simple_adversary_v3.parallel_env(render_mode = "human",N= 2, max_cycles = NUM_STEP, continuous_actions = True)


class CustomEnv(ParallelEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents=3, width=200, height=200, num_obstacles=5):
        super().__init__()
        self.num_agents = num_agents
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.obs_radius = 10
        self.agents = ["agent_" + str(i) for i in range(self.num_agents)]
        self.agent_radius = 2
        # self._agent_selector = agent_selector(self.agents)
        self.action_spaces = {agent: Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32) for agent in self.agents}
        self.observation_spaces = {agent: Box(low=0, high=max(self.width, self.height), shape=(4,), dtype=np.float32) for agent in self.agents}
        self.start_area = np.array([20,20])
        self.reset()

    def reset(self):
        self.agent_pos = {agent: np.random.rand(2) * self.start_area for agent in self.agents}
        self.obstacles = np.random.rand(self.num_obstacles, 2) * np.array([self.width, self.height])
        # self.target_pos = np.random.rand(2) * np.array([self.width, self.height])
        self.target_pos = {agent:np.random.rand(2)*np.array([200,200]) for agent in self.agents}
        # self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def step(self, actions):
        # agent = self._agent_selector.next()
        self._apply_action( actions)
        
        observations = {}
        rewards = {}
        dones = {}
        infos = {}

        for agent in self.agents:
            observations[agent] = self.observe(agent, self.target_pos[agent])
            rewards[agent] = self._calculate_reward(agent)
            dones[agent] = self._is_done(agent)
            infos[agent] = {}

        return observations, rewards, dones, infos

    def observe(self, agent, target_pos):
        # agent自身位置,与目标的距离和角度,与最近障碍物的距离和角度,与集群中其他智能体的距离和角度
        self_pos = self.agent_pos[agent]
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

    def _find_closest_obstacle(self, agent_pos):
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
        # This method should visualize the environment
        # You can use libraries like Pygame or Matplotlib for rendering
        pass

    def _apply_action(self, multi_action_pos):
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

    def _is_done(self, agent):
        # Check if the agent has reached the target
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