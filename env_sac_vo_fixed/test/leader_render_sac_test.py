import numpy as np
import torch
import torch.nn as nn
import os
import time
import torch.cuda
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from env_sac_vo.env_sac import CustomEnv
from env_sac_vo.circle_agent_sac import circle_agent, ReplayBuffer

RENDER_EPISODE_NUM = 1000
RENDER_NUM_STEP = 2000

NUM_REACH_GOAL = 0
NUM_COLLISION_SIDE = 0
NUM_COLLISION_OBS = 0
NUM_COLLISION_OBS_100 = 0
NUM_FAIL_REACH_GOAL = 0

scenario = "leader_sac_vo"
current_path = os.path.dirname(os.path.realpath(__file__))
agent_path = current_path + "/leader_model/better/" 

env = CustomEnv(delta=0.1)
env.leader_agent.sac_network.load_model(agent_path, scenario)
for episode_i in range(RENDER_EPISODE_NUM):
    state, done = env.reset()
    target_distance = np.linalg.norm(np.array(env.leader_agent.pos)- np.array(env.leader_target_pos))
    print("rendering episode ", episode_i," ==========================")
    
    for step_i in range (RENDER_NUM_STEP):
        # env.render(display_time=0.1)
        leader_action = env.leader_agent.sac_network.take_action(state)
        noise = np.random.normal(0, 0.05, size=leader_action.shape)

        noisy_action = leader_action + noise
        noisy_action = np.clip(noisy_action, -1, 1)

        # print("leader_action : ", leader_action)
        last_obs_distance = {}
        for obs_id, obs in env.obstacles.items():
            last_obs_distance[obs_id] = np.linalg.norm(np.array(env.leader_agent.pos) - np.array([obs.pos_x, obs.pos_y]))
        next_state, reward, done, target = env.step(leader_action, step_i, target_distance,last_obs_distance)
        
        if env.leader_agent.done and not env.leader_agent.target:
            if env.leader_agent.pos[0] < env.width * 0.1 or (env.width - env.leader_agent.pos[0]) < env.width * 0.1 or env.leader_agent.pos[1] < env.height * 0.1 or (env.height - env.leader_agent.pos[1]) < env.height * 0.1:
                NUM_COLLISION_SIDE += 1
                print(f"\033[91mxxxxxxxxxxxxxxxxxx  COLLISION SIDE xxxxxxxxxxxxxxxxxxx step{step_i} episode_reward : {reward:.2f}\033[0m")
                break
            else :
                if step_i <= 100:
                    NUM_COLLISION_OBS_100 +=1
                    print(f"\033[93mxxxxxxxxxxxxxxxxxx  COLLISION OBSTACLE <100 xxxxxxxxxxxxxxxxxxx step{step_i} reward : {reward:.2f}\033[0m")
                    break
                else:
                    NUM_COLLISION_OBS +=1
                    print(f"\033[93mxxxxxxxxxxxxxxxxxx  COLLISION OBSTACLE xxxxxxxxxxxxxxxxxxx step{step_i} reward : {reward:.2f}\033[0m")
                    break
        elif env.leader_agent.done and env.leader_agent.target:
            print(f"\033[92m******************** REACH GOAL ********************step{step_i} reward : {reward:.2f}\033[0m")
            NUM_REACH_GOAL += 1
            break
        state = next_state
    # env.render_close()

NUM_FAIL_REACH_GOAL = RENDER_EPISODE_NUM - NUM_COLLISION_OBS - NUM_COLLISION_SIDE - NUM_REACH_GOAL - NUM_COLLISION_OBS_100
print(f"测试结果为：到达目标 {NUM_REACH_GOAL} 次， 碰撞边缘 {NUM_COLLISION_SIDE} 次， 碰撞边缘但是初始距离小 {NUM_COLLISION_OBS_100} 次，碰撞障碍 {NUM_COLLISION_OBS} 次，未到达目标 {NUM_FAIL_REACH_GOAL} 次")
