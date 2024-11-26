import numpy as np
import torch
import torch.nn as nn
import os
import time
import torch.cuda
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env_formation.env_formation_single import CustomEnv
from env_formation.circle_agent_sac import circle_agent, ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device:{device}")

RENDER_EPISODE_NUM = 5
RENDER_NUM_STEP = 1000

scenario = "formation_sac_vo"
current_path = os.path.dirname(os.path.realpath(__file__))
agent_path = current_path + "/leader_model/"
better_path = current_path + "/leader_model/better/"
follower_path = current_path + "/follower_model/"
follower_better_path = current_path + "/follower_model/better/"

for episode_i in range(RENDER_EPISODE_NUM):
    env = CustomEnv(delta=0.1)
    # env.leader_agent.sac_network.load_model(better_path, scenario)
    env.leader_agent.sac_network.load_model(agent_path, scenario)
    # env.SAC.load_model(follower_better_path, scenario)
    env.SAC.load_model(follower_path, scenario)

    leader_state, leader_done = env.reset()
    target_distance = np.linalg.norm(np.array(env.leader_agent.pos) - np.array(env.leader_target_pos))
    print("rendering episode ", episode_i, "=============")

    for step_i in range (RENDER_NUM_STEP):
        # time.sleep(0.5)
        # print("==============================")
        env.render(display_time = 0.01)
        leader_action = env.leader_agent.sac_network.take_action(leader_state)
        leader_noisy_action = leader_action + np.random.normal(0, 0.05, size=leader_action.shape)
        leader_noisy_action = np.clip(leader_noisy_action, -1, 1)

        leader_target_distance = np.linalg.norm(np.array(env.leader_agent.pos)- np.array(env.leader_target_pos))
        last_obs_distance = {}
        for obs_id, obs in env.obstacles.items():
            last_obs_distance[obs_id] = np.linalg.norm(np.array(env.leader_agent.pos) - np.array([obs.pos_x, obs.pos_y]))

        last_follower_obs_distances = []
        last_follower_goal_distances = []
        for i in range(env.follower_uav_num):
            last_goal_dis = np.linalg.norm(np.array(env.follower_uavs[f"follower_{i}"].pos) - np.array(env.leader_agent.pos) - np.array(env.formation_pos[i]))
            last_follower_goal_distances.append(last_goal_dis)
            
            last_follower_obs_distance = {}
            for obs_id, obs in env.obstacles.items():
                last_follower_obs_distance[obs_id] = np.linalg.norm(np.array(env.follower_uavs[f"follower_{i}"].pos) - np.array([obs.pos_x, obs.pos_y]))
            last_follower_obs_distances.append(last_follower_obs_distance)

        # 跟随者的动作
        follower_actions = []
        follower_observations = []
        for i in range (env.follower_uav_num):
            # print(env.follower_uavs[f"follower_{i}"].observation.shape)
            follower_action = env.SAC.take_action(env.follower_uavs[f"follower_{i}"].observation)
            follower_observations. extend(env.follower_uavs[f"follower_{i}"].observation)
            noisy_follower_action = follower_action + np.random.normal(0, 0.2, size=follower_action.shape)
            noisy_follower_action = np.clip(noisy_follower_action, -1, 1)
            follower_actions.extend(noisy_follower_action)

        leader_next_state, reward, done, target, \
            next_follower_observations, follower_reward, follower_done = env.step(leader_action = leader_noisy_action,
                                                                                                follower_actions = follower_actions,
                                                                                                last_distance=leader_target_distance,
                                                                                                last_obs_distance=last_obs_distance,
                                                                                                last_follower_obs_distance=last_follower_obs_distances,
                                                                                                last_follower_goal_distance=last_follower_goal_distances)
        # 跟随者经验回放
        for i in range (env.follower_uav_num):
            follower_S = follower_observations[i]
            follower_A = np.array(follower_actions[2*i : 2*i+2])
            follower_R = np.array(follower_reward[i])
            follower_NS = next_follower_observations[i]
            follower_done_ = follower_done[i]
            # env.SAC.replay_buffer.add(state=follower_S, action=follower_A, reward=follower_R, next_state=follower_NS, done=follower_done_)
            # print(f"uav{i} state : {follower_S} ")
            # print(f"uav{i} action : {follower_A} ")
            # print(f"uav{i} reward : {follower_R} ")
            # print(f"uav{i} next_state : {follower_NS} ")
            # print(f"uav{i} done : {follower_done_} ")
            # print("=============================")


        if env.leader_agent.done and not env.leader_agent.target:
            print(f"xxxxxxxxxxxxxxxxxx  COLLISION  xxxxxxxxxxxxxxxxxxx step{step_i} reward : {reward}")
            break
        elif env.leader_agent.done and env.leader_agent.target:
            print(f"******************** REACH GOAL ********************step{step_i} reward : {reward}")
            break
        if follower_done :
            follower_break_flag = False
            for i in range(env.follower_uav_num):
                if env.follower_uavs[f"follower_{i}"].obs_done == True:
                    print(f"\033[93mxxxxxxxxxxxxxxxxxx  FOLLOWER OBSTICLE COLLISION xxxxxxxxxxxxxxxxxxx step{step_i} \033[0m")
                    follower_break_flag = True
                    break
                if env.follower_uavs[f"follower_{i}"].side_done == True:
                    print(f"\033[91mxxxxxxxxxxxxxxxxxx  FOLLOWER SIDE COLLISION xxxxxxxxxxxxxxxxxxx step{step_i} \033[0m")
                    follower_break_flag = True
                    break
                if env.follower_uavs[f"follower_{i}"].uav_done == True:
                    print(f"\033[34mxxxxxxxxxxxxxxxxxx  FOLLOWER UAV COLLISION xxxxxxxxxxxxxxxxxxx step{step_i} \033[0m")
                    follower_break_flag = True
                    break
                if env.follower_uavs[f"follower_{i}"].formation_done == True:
                    print(f"\033[35mxxxxxxxxxxxxxxxxxx  FOLLOWER FORMATION DONE xxxxxxxxxxxxxxxxxxx step{step_i} \033[0m")
                    follower_break_flag = True
                    break
            if follower_break_flag == True:
                break
        leader_state = leader_next_state
    env.render_close()