import numpy as np
import torch
import torch.nn as nn
import os
import time
import torch.cuda
import sys
from numpy import inf
import torch.nn.functional as F
from numpy import sqrt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from env_sac.env_sac import CustomEnv
from env_sac.circle_agent_sac import circle_agent, ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device:{device}")

NUM_EPISODE = 3000
NUM_STEP = 500
MEMORY_SIZE = 100000
BATCH_SIZE = 512
TARGET_UPDATE_INTERVAL= NUM_STEP *4


env = CustomEnv(delta=0.1)
env.reset()

scenario = "leader_sac"
current_path = os.path.dirname(os.path.realpath(__file__))
agent_path = current_path + "/leader_model/"

timestamp = time.strftime("%Y%m%d%H%M%S")

# main train loop
episode_highest_reward = -inf

for episode_i in range(NUM_EPISODE):

    state, done = env.reset()
    target_distance = np.linalg.norm(np.array(env.leader_agent.pos), np.array(env.leader_target_pos))

    if os.path.exists(agent_path) and episode_i > NUM_EPISODE*(5/6):
        env.leader_agent.sac_network.load_model(agent_path, scenario)

    if episode_i > 0:
        print(f"episode {episode_i} =============== HIGHEST REWARD : {leader_highest_step_reward:.2f} ========================")
    
    episode_reward = 0
    reward_list =[]
    for step_i in range(NUM_STEP):
        while not done:
            if episode_i and step_i == 0 :
                leader_highest_step_reward = -inf
            
            total_step = episode_i *NUM_STEP +step_i

            leader_action = env.leader_agent.sac_network.take_action(state)

            next_state, reward, done, target = env.step(action=leader_action,
                                                        num_step=total_step,
                                                        target_distance=target_distance)
            episode_reward = episode_reward * 0.9 + reward
            # 保存每个回合return
            reward_list.append(episode_reward)
            env.leader_agent.replay_buffer.add(state=state, action=leader_action, reward=reward, next_state=next_state,done=done)
            
            state = next_state
            
            current_memo_size = min(MEMORY_SIZE, total_step)
            batch_flag = False   
            if current_memo_size >= BATCH_SIZE*5:    
                batch_flag = True
            else:
                batch_flag = False
                continue
            if(total_step +1)% TARGET_UPDATE_INTERVAL == 0 and batch_flag == True:
                s, a, r, ns, d = env.leader_agent.replay_buffer.sample(batch_size=BATCH_SIZE)
                transition_dict = {'states': s,
                               'actions': a,
                               'rewards': r,
                               'next_states': ns,
                               'dones': d}
                env.leader_agent.sac_network.update(transition_dict=transition_dict)

            