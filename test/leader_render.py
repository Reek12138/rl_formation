import numpy as np
import torch
import torch.nn as nn
import os
import time
import torch.cuda
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from env.env_base import CustomEnv
from env.circle_agent import circle_agent, ReplayBuffer

RENDER_EPISODE_NUM = 5
RENDER_NUM_STEP = 300

env = CustomEnv(delta=0.5)
NUM_AGENT = env.num_agents
MODE = "d"


scenario = "my_env"
current_path = os.path.dirname(os.path.realpath(__file__))
agent_path = current_path + "/leader_models/" + scenario + '/'
timestamp = time.strftime("%Y%m%d%H%M%S")

env.leader_agent.actor.load_checkpoint(f"{agent_path}" + f"leader_agent_actor_{scenario}.pth")
for episode_i in range(RENDER_EPISODE_NUM):
    multi_obs, infos = env.reset()
    print("rendering episode ", episode_i," ==========================")

    for step_i in range(RENDER_NUM_STEP):
        env.render(display_time=0.1)
        leader_action = env.leader_agent.get_action(multi_obs["leader_agent"], MODE)
        print("leader_action : ", leader_action)

        multi_actions = {}#follower_action
        for agent_id, agent in env.follower_agents.items():
            multi_actions[agent_id] = [0,0]
                
        multi_next_obs, multi_rewards, multi_done, infos = env.step(leader_action=leader_action, follower_actions=multi_actions)
        multi_obs = multi_next_obs
    env.render_close()
