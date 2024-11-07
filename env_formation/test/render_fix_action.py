import numpy as np
import torch
import torch.nn as nn
import os
import time
import torch.cuda
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env_formation.env_formation import CustomEnv
from env_formation.circle_agent_sac import circle_agent, ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device:{device}")

RENDER_EPISODE_NUM = 5
RENDER_NUM_STEP = 1000

scenario = "formation_sac_vo"
current_path = os.path.dirname(os.path.realpath(__file__))
agent_path = current_path + "/leader_model/"
better_path = current_path + "/leader_model/better/"
follower_path = current_path + "/follower_model"
follower_better_path = current_path + "/follower_model/better/"

for episode_i in range(RENDER_EPISODE_NUM):
    env = CustomEnv(delta=0.1)
    # env.leader_agent.sac_network.load_model(better_path, scenario)
    # env.MASAC.load_model(follower_better_path, scenario)

    leader_state, leader_done = env.reset()
    target_distance = np.linalg.norm(np.array(env.leader_agent.pos) - np.array(env.leader_target_pos))
    print("rendering episode ", episode_i, "=============")

    for step_i in range (RENDER_NUM_STEP):
        env.render(display_time = 0.1)
        env.leader_agent.orientation = np.pi/4

        leader_noisy_action = [1.0, 0.0]

        leader_target_distance = np.linalg.norm(np.array(env.leader_agent.pos)- np.array(env.leader_target_pos))
        last_obs_distance = {}
        for obs_id, obs in env.obstacles.items():
            last_obs_distance[obs_id] = np.linalg.norm(np.array(env.leader_agent.pos) - np.array([obs.pos_x, obs.pos_y]))

        follower_actions = []
        for i in range (env.follower_uav_num):
            follower_action_i = [1.414, 1.414]
            follower_actions.extend(follower_action_i)

        leader_next_state, reward, done, target, \
            next_follower_observations, follower_reward, follower_done = env.step(leader_action = leader_noisy_action,
                                                                                                follower_actions = follower_actions,
                                                                                                last_distance=leader_target_distance,
                                                                                                last_obs_distance=last_obs_distance)
        if env.leader_agent.done and not env.leader_agent.target:
            print(f"xxxxxxxxxxxxxxxxxx  COLLISION  xxxxxxxxxxxxxxxxxxx step{step_i} reward : {reward}")
            break
        elif follower_done:
            print(f"xxxxxxxxxxxxxxxxxx FOLLOWER COLLISION  xxxxxxxxxxxxxxxxxxx step{step_i} reward : {reward}")
            break
        elif env.leader_agent.done and env.leader_agent.target:
            print(f"******************** REACH GOAL ********************step{step_i} reward : {reward}")
            break
        leader_state = leader_next_state
    env.render_close()