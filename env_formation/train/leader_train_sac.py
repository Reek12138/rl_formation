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
from math import sin, cos, tan, pi, sqrt, log

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from env_formation.env_formation import CustomEnv
from env_formation.circle_agent_sac import circle_agent, ReplayBuffer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device:{device}")

NUM_EPISODE = 90002
NUM_STEP = 3000
MEMORY_SIZE = 100000
# BATCH_SIZE = 512
BATCH_SIZE = 1024
TARGET_UPDATE_INTERVAL= 20
RENDER_FREQUENCE = 500
RENDER_NUM_EPISODE = 200
RENDER_NUM_STEP = 1000
BREAK_FLAG = False


env = CustomEnv(delta=0.1)
env.reset()

scenario = "formation_sac_vo"
current_path = os.path.dirname(os.path.realpath(__file__))
agent_path = current_path + "/leader_model/"
better_path = current_path + "/leader_model/better/"
follower_path = current_path + "/follower_model"
follower_better_path = current_path + "/follower_model/better/"

timestamp = time.strftime("%Y%m%d%H%M%S")

# main train loop
last_episode_reward = -inf
ave_highest_reward = -inf
highest_num_reach_goal = 0

last_episode_follower_reward = -inf

env.leader_agent.replay_buffer.clear()
env.MASAC.replay_buffer.clear()
for episode_i in range(NUM_EPISODE):
    if BREAK_FLAG == True:
        break
    leader_state, done = env.reset()
    # print(done)
    
    target_distance = np.linalg.norm(np.array(env.leader_agent.pos) - np.array(env.leader_target_pos))
    # print(env.leader_target_pos, env.leader_agent.pos)
    # if os.path.exists(agent_path) and episode_i > NUM_EPISODE*(2/3):
    #     env.leader_agent.sac_network.load_model(better_path, scenario)

    if episode_i > 0:
        print(f"episode {episode_i -1}   ===========    LAST EPISODE REWARD : {last_episode_reward:.2f} STEP : {num_step} REWARD : {leader_highest_step_reward:.2f}                ")
    
    episode_reward = 0
    reward_list =[]
    leader_highest_step_reward = -inf

    follower_episode_reward = 0
    follower_reward_list = []
    follower_highest_step_reward = -inf

    num_step = 0
    for step_i in range(NUM_STEP):
        num_step += 1
        # while not done:
        if episode_i == 0 and step_i == 0 :
            leader_highest_step_reward = -inf
            follower_highest_step_reward = -inf

        
        total_step = episode_i *NUM_STEP +step_i
        # print(state)

        # 领航者的动作
        leader_action = env.leader_agent.sac_network.take_action(leader_state)
        # print(leader_action)
        # leader_action = leader_action_[:2]
        noise = np.random.normal(0, 0.2, size=leader_action.shape)
        noisy_action = leader_action + noise
        noisy_action = np.clip(noisy_action, -1, 1)
        action_in =[noisy_action[0] + 1,
                    noisy_action[1] * (np.pi/2)]
        
        # 与上一次的距离
        last_distance = np.linalg.norm(np.array(env.leader_agent.pos) - np.array(env.leader_target_pos))
        last_obs_distance = {}
        for obs_id, obs in env.obstacles.items():
            last_obs_distance[obs_id] = np.linalg.norm(np.array(env.leader_agent.pos) - np.array([obs.pos_x, obs.pos_y]))
        # print(leader_action_noise)
        
        last_follower_obs_distances = []
        for i in range(env.follower_uav_num):
            last_follower_obs_distance = {}
            for obs_id, obs in env.obstacles.items():
                last_follower_obs_distance[obs_id] = np.linalg.norm(np.array(env.follower_uavs[f"follower_{i}"].pos) - np.array([obs.pos_x, obs.pos_y]))
            last_follower_obs_distances.append(last_follower_obs_distance)

        # 跟随者的动作
        follower_actions = []
        follower_observations = []
        for i in range (env.follower_uav_num):
            # print(env.follower_uavs[f"follower_{i}"].observation.shape)
            follower_action = env.MASAC.take_action(env.follower_uavs[f"follower_{i}"].observation)
            follower_observations. extend(env.follower_uavs[f"follower_{i}"].observation)
            noisy_follower_action = follower_action + np.random.normal(0, 0.2, size=follower_action.shape)
            noisy_follower_action = np.clip(noisy_follower_action, -1, 1)
            follower_actions.extend(noisy_follower_action)
        
        

        
        leader_next_state, reward, done, target, \
            next_follower_observations, follower_reward, follower_done = env.step(leader_action = noisy_action,
                                                                                                follower_actions = follower_actions,
                                                                                                last_distance=last_distance,
                                                                                                last_obs_distance=last_obs_distance,
                                                                                                last_follower_obs_distance=last_follower_obs_distances)
        
        # follower_A = np.concatenate(follower_actions, axis=0)
        # follower_S = np.concatenate(follower_observations, axis=0)
        # follower_NS = np.concatenate(next_follower_observations, axis=0)
        # follower_R = sum(follower_reward)/ env.follower_uav_num
        follower_A = np.array(follower_actions)
        follower_S = np.array(follower_observations)
        follower_NS = np.array(next_follower_observations)
        follower_R = sum(follower_reward)/ env.follower_uav_num


        if reward > leader_highest_step_reward:
            # print(f"highest step reward update {reward:.2f} at episode {episode_i} step  {step_i}")
            leader_highest_step_reward = reward
        episode_reward = episode_reward * 0.9 + reward
        # 保存每个回合return
        reward_list.append(episode_reward)
        # print (state, leader_action)

        if follower_R >follower_highest_step_reward:
            follower_highest_step_reward = follower_R
        follower_episode_reward = follower_episode_reward * 0.9 + follower_R
        follower_reward_list.append(follower_episode_reward)

        # 结束
        if step_i == NUM_STEP:
            done = True
            follower_done = True

        # 领航者经验回放
        env.leader_agent.replay_buffer.add(state=leader_state, action=leader_action, reward=reward, next_state=leader_next_state,done=done)
        # 跟随者经验回放
        env.MASAC.replay_buffer.add(state=follower_S, action=follower_A, reward=follower_R, next_state=follower_NS, done=follower_done)

        leader_state = leader_next_state
        
#====== 网络更新========================================================================================
        current_memo_size = min(MEMORY_SIZE, total_step)
        batch_flag = current_memo_size >= BATCH_SIZE * 5
        if(total_step +1)% TARGET_UPDATE_INTERVAL == 0 and batch_flag == True and episode_i <= NUM_EPISODE/3:
            if env.leader_agent.replay_buffer.size() >= BATCH_SIZE:
                # 领航者的网络更新
                s, a, r, ns, d = env.leader_agent.replay_buffer.sample(batch_size=BATCH_SIZE)
                transition_dict = {'states': s,
                                'actions': a,
                                'rewards': r,
                                'next_states': ns,
                                'dones': d}
                env.leader_agent.sac_network.update(transition_dict=transition_dict)

                # 跟随者的网络更新
                fs, fa, fr, fns, fd = env.MASAC.replay_buffer.sample(batch_size=BATCH_SIZE)
                f_transition_dict = {'states' : fs,
                                    'actions' : fa,
                                    'rewards' : fr,
                                    'next_states' : fns,
                                    'dones' : fd}
                env.MASAC.update(transition_dict= f_transition_dict)
            
        
# ======碰撞检测========================================================================================
        # 如果领航者碰撞
        if env.leader_agent.done and not env.leader_agent.target:
            if env.leader_agent.pos[0] < env.width * 0.1 \
                or (env.width - env.leader_agent.pos[0]) < env.width * 0.1 \
                or env.leader_agent.pos[1] < env.height * 0.1 \
                or (env.height - env.leader_agent.pos[1]) < env.height * 0.1:
                print(f"\033[91mxxxxxxxxxxxxxxxxxx  COLLISION SIDE xxxxxxxxxxxxxxxxxxx step{step_i} episode_reward : {episode_reward:.2f}\033[0m")
                # print(f"xxxxxxxxxxxxxxxxxx  COLLISION SIDE xxxxxxxxxxxxxxxxxxx step{step_i} reward : {reward}")
            else :
                print(f"\033[93mxxxxxxxxxxxxxxxxxx  COLLISION OBSTACLE xxxxxxxxxxxxxxxxxxx step{step_i} reward : {episode_reward:.2f}\033[0m")
            break
        elif env.leader_agent.done and env.leader_agent.target:
            print(f"\033[92m******************** REACH GOAL ********************step{step_i} reward : {episode_reward:.2f}\033[0m")
            break

        # 如果跟随者碰撞
        if follower_done :
            for i in range(env.follower_uav_num):
                if env.follower_uavs[f"follower_{i}"].obs_done == True:
                    print(f"\033[93mxxxxxxxxxxxxxxxxxx  FOLLOWER OBSTICLE COLLISION xxxxxxxxxxxxxxxxxxx step{step_i} reward : {follower_episode_reward:.2f}\033[0m")
                    break
                if env.follower_uavs[f"follower_{i}"].side_done == True:
                    print(f"\033[91mxxxxxxxxxxxxxxxxxx  FOLLOWER SIDE COLLISION xxxxxxxxxxxxxxxxxxx step{step_i} reward : {follower_episode_reward:.2f}\033[0m")
                    break
                if env.follower_uavs[f"follower_{i}"].uav_done == True:
                    print(f"\033[34mxxxxxxxxxxxxxxxxxx  FOLLOWER UAV COLLISION xxxxxxxxxxxxxxxxxxx step{step_i} reward : {follower_episode_reward:.2f}\033[0m")
                    break
            break

#====== 每个episode结束后更新一次=====================================================================
    if env.leader_agent.replay_buffer.size() >= BATCH_SIZE:
        s, a, r, ns, d = env.leader_agent.replay_buffer.sample(batch_size=BATCH_SIZE)
        transition_dict = {'states': s,
                        'actions': a,
                        'rewards': r,
                        'next_states': ns,
                        'dones': d}
        env.leader_agent.sac_network.update(transition_dict=transition_dict)

        if not os.path.exists(agent_path):
            os.makedirs(agent_path)
        env.leader_agent.sac_network.save_model(agent_path, scenario)

        fs, fa, fr, fns, fd = env.MASAC.replay_buffer.sample(batch_size=BATCH_SIZE)
        f_transition_dict = {'states' : fs,
                            'actions' : fa,
                            'rewards' : fr,
                            'next_states' : fns,
                            'dones' : fd}
        env.MASAC.update(transition_dict= f_transition_dict)

        if not os.path.exists(follower_path):
            os.makedirs(follower_path)
        env.MASAC.save_model(follower_path, scenario)



    last_episode_reward = episode_reward
    last_episode_follower_reward = follower_episode_reward

    
    # 验证
    if episode_i > 0 and episode_i % RENDER_FREQUENCE == 0:
    # if episode_i > NUM_EPISODE/3 and episode_i % RENDER_FREQUENCE == 0:
        env = CustomEnv(delta=0.1)
        num_reach_goal = 0
        num_collision_side = 0
        num_collision_obstacle = 0
        num_follower_collision = 0
        print("======================验证中=========================")

        for test_episode_i in range (RENDER_NUM_EPISODE):
            

            env.leader_agent.sac_network.load_model(agent_path, scenario)
            env.MASAC.load_model(follower_path, scenario)
            target_distance = np.linalg.norm(np.array(env.leader_agent.pos) - np.array(env.leader_target_pos))

            leader_state, done = env.reset()

            for test_step_i in range (RENDER_NUM_STEP):
                
                # env.render(display_time=0.1)
                leader_action = env.leader_agent.sac_network.take_action(leader_state)
                noise = np.random.normal(0, 0.2, size=leader_action.shape)
                noisy_action = leader_action + noise
                noisy_action = np.clip(noisy_action, -1, 1)
                action_in =[noisy_action[0] + 1,
                            noisy_action[1] * (np.pi/2)]

                
                # print("leader_action : ",leader_action)
                last_distance = np.linalg.norm(np.array(env.leader_agent.pos) - np.array(env.leader_target_pos))
                last_obs_distance = {}
                for obs_id, obs in env.obstacles.items():
                    last_obs_distance[obs_id] = np.linalg.norm(np.array(env.leader_agent.pos) - np.array([obs.pos_x, obs.pos_y]))

                last_follower_obs_distances = []
                for i in range(env.follower_uav_num):
                    last_follower_obs_distance = {}
                    for obs_id, obs in env.obstacles.items():
                        last_follower_obs_distance[obs_id] = np.linalg.norm(np.array(env.follower_uavs[f"follower_{i}"].pos) - np.array([obs.pos_x, obs.pos_y]))
                    last_follower_obs_distances.append(last_follower_obs_distance)

                # 跟随者的动作
                follower_actions = []
                follower_observations = []
                for i in range (env.follower_uav_num):
                    follower_action = env.MASAC.take_action(env.follower_uavs[f"follower_{i}"].observation)
                    follower_observations. extend(env.follower_uavs[f"follower_{i}"].observation)
                    noisy_follower_action = follower_action + np.random.normal(0, 0.2, size=follower_action.shape)
                    noisy_follower_action = np.clip(noisy_follower_action, -1, 1)
                    follower_actions.extend(noisy_follower_action)

                leader_next_state, reward, done, target, \
                    next_follower_observations, follower_reward, follower_done = env.step(leader_action = noisy_action,
                                                                                                follower_actions = follower_actions,
                                                                                                last_distance=last_distance,
                                                                                                last_obs_distance=last_obs_distance,
                                                                                                last_follower_obs_distance=last_follower_obs_distances)
                
                leader_state = leader_next_state

                if env.leader_agent.done and not env.leader_agent.target:
                    if env.leader_agent.pos[0] < env.width * 0.1 or (env.width - env.leader_agent.pos[0]) < env.width * 0.1 or env.leader_agent.pos[1] < env.height * 0.1 or (env.height - env.leader_agent.pos[1]) < env.height * 0.1:
                        num_collision_side += 1
                        print(f"\033[91mxxxxxxxxxxxxxxxxxx  COLLISION SIDE xxxxxxxxxxxxxxxxxxx step{test_step_i} episode_reward : {episode_reward:.2f}\033[0m")
                    else :
                        num_collision_obstacle+=1
                        print(f"\033[93mxxxxxxxxxxxxxxxxxx  COLLISION OBSTACLE xxxxxxxxxxxxxxxxxxx step{test_step_i} reward : {episode_reward:.2f}\033[0m")
                    break
                elif env.leader_agent.done and env.leader_agent.target:
                    num_reach_goal+=1
                    print(f"\033[92m******************** REACH GOAL ********************step{test_step_i} reward : {episode_reward:.2f}\033[0m")
                    break
                # 如果跟随者碰撞
                if follower_done :
                    print(f"\033[34mxxxxxxxxxxxxxxxxxx  FOLLOWER UAV COLLISION xxxxxxxxxxxxxxxxxxx step{step_i} reward : {follower_episode_reward:.2f}\033[0m")
                    num_follower_collision+=1
                    # for i in range(env.follower_uav_num):
                    #     if env.follower_uavs[f"follower_{i}"].obs_done == True:
                    #         print(f"\033[93mxxxxxxxxxxxxxxxxxx  FOLLOWER OBSTICLE COLLISION xxxxxxxxxxxxxxxxxxx step{step_i} reward : {follower_episode_reward:.2f}\033[0m")
                    #         num_follower_collision+=1
                    #         break
                    #     if env.follower_uavs[f"follower_{i}"].side_done == True:
                    #         print(f"\033[91mxxxxxxxxxxxxxxxxxx  FOLLOWER SIDE COLLISION xxxxxxxxxxxxxxxxxxx step{step_i} reward : {follower_episode_reward:.2f}\033[0m")
                    #         num_follower_collision+=1
                    #         break
                    #     if env.follower_uavs[f"follower_{i}"].uav_done == True:
                    #         print(f"\033[34mxxxxxxxxxxxxxxxxxx  FOLLOWER UAV COLLISION xxxxxxxxxxxxxxxxxxx step{step_i} reward : {follower_episode_reward:.2f}\033[0m")
                    #         num_follower_collision+=1
                    #         break
                    break
                    

            # env.render_close()

        print(f"\033[95mreach goal : {num_reach_goal}, collision_side : {num_collision_side}, collision obstacle : {num_collision_obstacle}, follower collision : {num_follower_collision} 最高  : {highest_num_reach_goal}\033[0m")
        if num_reach_goal >= highest_num_reach_goal:
            if not os.path.exists(better_path):
                os.makedirs(better_path)
            env.leader_agent.sac_network.save_model(better_path, scenario)

            if not os.path.exists(follower_path):
                os.makedirs(follower_path)
            env.MASAC.save_model(follower_path, scenario)

            print("--------------------------更好的参数-----------------")
            if num_reach_goal == RENDER_NUM_EPISODE:
                BREAK_FLAG = True

            highest_num_reach_goal = num_reach_goal

        if  episode_i > NUM_EPISODE*(2/3):
            env.leader_agent.sac_network.load_model(better_path, scenario)
            env.MASAC.load_model(follower_better_path, scenario)
        else:
            env.leader_agent.sac_network.load_model(agent_path, scenario)
            env.MASAC.load_model(follower_path, scenario)
