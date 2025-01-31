import numpy as np
import torch
import torch.nn as nn
import os
import time
import torch.cuda
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from env_sac_vo.env_sac_vo.env_sac import CustomEnv
from env_sac_vo.env_sac_vo.circle_agent_sac import circle_agent, ReplayBuffer


RENDER_EPISODE_NUM = 5
RENDER_NUM_STEP = 1000

scenario = "leader_sac_vo"
current_path = os.path.dirname(os.path.realpath(__file__))
agent_path = current_path + "/leader_model/better/" 
# agent_path = current_path + "/leader_model/" 

# env = CustomEnv(delta=0.1)
# env.leader_agent.sac_network.load_model(agent_path, scenario)
for episode_i in range(RENDER_EPISODE_NUM):
    env = CustomEnv(delta=0.1)
    env.leader_agent.sac_network.load_model(agent_path, scenario)

    state, done = env.reset()
    print("rendering episode ", episode_i," ==========================")
    
    for step_i in range (RENDER_NUM_STEP):
        env.render(display_time=0.1)
        leader_action = env.leader_agent.sac_network.take_action(state)
        noise = np.random.normal(0, 0.05, size=leader_action.shape)

        noisy_action = leader_action + noise
        noisy_action = np.clip(noisy_action, -1, 1)

        # print("leader_action : ", leader_action)
        target_distance = np.linalg.norm(np.array(env.leader_agent.pos)- np.array(env.leader_target_pos))
        last_obs_distance = {}
        for obs_id, obs in env.obstacles.items():
            last_obs_distance[obs_id] = np.linalg.norm(np.array(env.leader_agent.pos) - np.array([obs.pos_x, obs.pos_y]))
        next_state, reward, done, target = env.step(leader_action, step_i, target_distance,last_obs_distance)
        
        if env.leader_agent.done and not env.leader_agent.target:
            print(f"xxxxxxxxxxxxxxxxxx  COLLISION  xxxxxxxxxxxxxxxxxxx step{step_i} reward : {reward}")
            break
        elif env.leader_agent.done and env.leader_agent.target:
            print(f"******************** REACH GOAL ********************step{step_i} reward : {reward}")
            break
        state = next_state
    env.render_close()
