import numpy as np
import torch
import torch.nn as nn
import os
import time
import torch.cuda
import sys
from numpy import inf
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from env.env_base import CustomEnv
from env.circle_agent import circle_agent, ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device:{device}")

def multi_obs_to_state(multi_obs):
    state = np.array([])
    for agent_id, agent_obs in multi_obs.items():
        if agent_id != "leader_agent":
            state = np.concatenate([state, agent_obs])
    return state

NUM_EPISODE = 9000
NUM_STEP = 2000
MEMORY_SIZE = 100000
BATCH_SIZE = 512
TARGET_UPDATE_INTERVAL=100
LR_ACTOR = 0.01
LR_CRITIC = 0.01
HIDDEN_DIM = 64
GAMMA = 0.99
TAU = 0.01
# LEADER_MAX_LINEAR_VEL = 10
MODE = "a"
RENDER_FREQUENCE = 500
RENDER_NUM_STEP = 100

env = CustomEnv(delta=0.5)
multi_obs, infos = env.reset()
#这里要根据环境修改
NUM_AGENT = env.num_agents
agent_name_list = env.follower_agents.keys()


scenario = "my_env"
current_path = os.path.dirname(os.path.realpath(__file__))
agent_path = current_path + "/leader_models/" + scenario + '/'
timestamp = time.strftime("%Y%m%d%H%M%S")

agents = [ ]
for agent in env.follower_agents.values():
    agents.append(agent)

# 2. main training loop
episode_highest_reward = -inf    
for episode_i in range(NUM_EPISODE):
    # print("reset2==========================")
    

    multi_obs, infos = env.reset()
    if os.path.exists(agent_path) and episode_i > NUM_EPISODE*(4/6):
        env.leader_agent.actor.load_checkpoint(f"{agent_path}" + f"leader_agent_actor_{scenario}.pth")
        env.leader_agent.target_actor.load_checkpoint(f"{agent_path}" + f"leader_agent_target_actor_{scenario}.pth")
        env.leader_agent.critic.load_checkpoint(f"{agent_path}" + f"leader_agent_critic_{scenario}.pth")
        env.leader_agent.target_critic.load_checkpoint(f"{agent_path}" + f"leader_agent_target_critic_{scenario}.pth")
    
    if NUM_EPISODE*(1/6) <= episode_i < NUM_EPISODE*(2/6) :
        MODE = "b"
    elif NUM_EPISODE*(2/6) <= episode_i < NUM_EPISODE*(4/6):
        MODE = "c"
    elif NUM_EPISODE*(4/6)<= episode_i:
        MODE = "d"

    # print(multi_obs)

    # leader_episode_reward = 0
    multi_done = {agent_name: False for agent_name in agent_name_list}
    multi_done["leader_agent"] = False
    # print(multi_done)

    
    if episode_i >0:
        print("episode",episode_i,"========================HIGHEST REWARD : ", leader_highest_reward)
    av_Q = 0
    max_Q = -inf
    av_loss = 0
    episode_reward = 0
    for step_i in range(NUM_STEP):
        if episode_i == 0 and step_i == 0:
            leader_highest_reward = -inf

        total_step = episode_i*NUM_STEP + step_i

        #2.1 collect actions from all agents 
        leader_action = env.leader_agent.get_action(multi_obs["leader_agent"], MODE)
        
        # print("action : ",leader_action, "observation : ", multi_obs["leader_agent"])

        multi_actions = {}#follower_action
        for agent_id, agent in env.follower_agents.items():
            multi_actions[agent_id]=[0,0]
            
        multi_next_obs, multi_rewards, multi_done, infos = env.step(leader_action=leader_action, follower_actions=multi_actions)
        
        episode_reward = multi_rewards["leader_agent"] + episode_reward*0.9

        if step_i >= NUM_STEP -1:
        # multi_done = {agnet_name :True for agnet_name in agent_name_list}
            env.leader_agent.done = True
            multi_done["leader_agent"] = env.leader_agent.done
            for agent_id, agent in env.follower_agents.items():
                agent.done = True
                multi_done[agent_id] = agent.done
        
        env.leader_agent.replay_buffer.add_memo(multi_obs["leader_agent"],
                                                multi_next_obs["leader_agent"],
                                                np.array(multi_obs["leader_agent"]),
                                                np.array(multi_next_obs["leader_agent"]),
                                                leader_action,
                                                multi_rewards["leader_agent"],
                                                multi_done["leader_agent"]
                                                )
        # 2.4.1 sample a batch of memories
        current_memo_size = min(MEMORY_SIZE, total_step+1)
        if current_memo_size < BATCH_SIZE:
            batch_idx = range(0, current_memo_size)
        else:
            batch_idx = np.random.choice(current_memo_size,  BATCH_SIZE, replace=False)        

        # update critic and actor 
        if(total_step +1)% TARGET_UPDATE_INTERVAL == 0:
        # if leader_episode_reward > leader_highest_reward:
        

            #leader_agent update
            leader_batch_obses, leader_batch_next_obses, leader_batch_states, leader_batch_next_state, leader_batch_actions, leader_batch_rewards, leader_batch_dones  = env.leader_agent.replay_buffer.sample(batch_idx)
            # single+batch 
            leader_batch_obses_tensor = torch.tensor(leader_batch_obses, dtype = torch.float).to(device)
            leader_batch_next_obses_tensor = torch.tensor(leader_batch_next_obses, dtype = torch.float).to(device)
            leader_batch_states_tensor = torch.tensor(leader_batch_states, dtype = torch.float).to(device)
            leader_batch_next_state_tensor = torch.tensor(leader_batch_next_state, dtype = torch.float).to(device)
            leader_batch_actions_tensor = torch.tensor(leader_batch_actions, dtype = torch.float).to(device)
            leader_batch_rewards_tensor = torch.tensor(leader_batch_rewards, dtype = torch.float).to(device)
            leader_batch_dones_tensor = torch.tensor(leader_batch_dones, dtype = torch.float).to(device)

            leader_next_action = env.leader_agent.target_actor(leader_batch_next_state_tensor)
            noise = torch.Tensor(leader_batch_actions).data.normal_(0,0.2).to(device)
            noise = noise.clamp(-0.5, 0.5)
            leader_next_action = (leader_next_action + noise).clamp(-1,1)

            target_q1, target_q2 = env.leader_agent.target_critic(leader_batch_next_state_tensor, leader_next_action)

            target_q = torch.min(target_q1, target_q2)
            av_Q += torch.min(target_q)
            max_Q = max(max_Q, torch.max(target_q))

            target_q = leader_batch_rewards_tensor + ((1-leader_batch_dones_tensor)*env.leader_agent.gamma*target_q).detach()

            current_q1, current_q2 = env.leader_agent.critic(leader_batch_states_tensor, leader_batch_actions_tensor)

            leader_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            env.leader_agent.critic.optimizer.zero_grad()
            leader_loss.backward()
            env.leader_agent.critic.optimizer.step()

            if(total_step+1)/2 ==0:
                actor_grad, _ = env.leader_agent.critic(leader_batch_states_tensor, env.leader_agent.actor(leader_batch_states_tensor))
                actor_grad = -actor_grad.mean()
                env.leader_agent.actor.optimizer.zero_grad()
                actor_grad.backward()
                env.leader_agent.actor.optimizer.step()

                # update target critic 
                for target_param, param in zip(env.leader_agent.target_critic.parameters(), env.leader_agent.critic.parameters()):
                    target_param.data.copy_(env.leader_agent.tau * param.data + (1.0 - env.leader_agent.tau) * target_param.data)
                # update target actor 
                for target_param, param in zip(env.leader_agent.target_actor.parameters(), env.leader_agent.actor.parameters()):
                    target_param.data.copy_(env.leader_agent.tau * param.data + (1.0 - env.leader_agent.tau) * target_param.data)
            
            av_loss += leader_loss


            flag = os.path.exists(agent_path)
            if not flag:
                os.makedirs(agent_path)
            # torch.save(env.leader_agent.actor.state_dict(), f"{agent_path}" + f"leader_agent_actor_{scenario}.pth")
            env.leader_agent.actor.save_checkpoint(f"{agent_path}" + f"leader_agent_actor_{scenario}.pth")
            env.leader_agent.target_actor.save_checkpoint(f"{agent_path}" + f"leader_agent_target_actor_{scenario}.pth")
            env.leader_agent.critic.save_checkpoint(f"{agent_path}" + f"leader_agent_critic_{scenario}.pth")
            env.leader_agent.target_critic.save_checkpoint(f"{agent_path}" + f"leader_agent_target_critic_{scenario}.pth")












            # ==========================================================================
            # #target actor output
            # leader_single_batch_next_actions = env.leader_agent.target_actor.forward(leader_batch_next_obses_tensor)

            # #actor output
            # leader_single_batch_online_action = env.leader_agent.actor.forward(leader_batch_obses_tensor)


            # # for agent_i in range(NUM_AGENT):
            # leader_critic_target_q = env.leader_agent.target_critic.forward(leader_batch_next_state_tensor, 
            #                                                                 leader_batch_actions_tensor.detach())
            # leader_y = (leader_batch_rewards_tensor + (1 - leader_batch_dones_tensor) * env.leader_agent.gamma * leader_critic_target_q).flatten()

            # leader_critic_q = env.leader_agent.critic.forward(leader_batch_states_tensor, leader_batch_actions_tensor.detach()).flatten()

            # leader_critic_loss = nn.MSELoss()(leader_y, leader_critic_q)
            # env.leader_agent.critic.optimizer.zero_grad()
            # leader_critic_loss.backward()
            # env.leader_agent.critic.optimizer.step()

            # leader_actor_loss = env.leader_agent.critic.forward(leader_batch_states_tensor,
            #                                                     leader_single_batch_online_action.detach()).flatten()
            # leader_actor_loss = -torch.mean(leader_actor_loss)
            # env.leader_agent.actor.optimizer.zero_grad()
            # leader_actor_loss.backward()
            # env.leader_agent.actor.optimizer.step()

            # # update target critic 
            # for target_param, param in zip(env.leader_agent.target_critic.parameters(), env.leader_agent.critic.parameters()):
            #     target_param.data.copy_(env.leader_agent.tau * param.data + (1.0 - env.leader_agent.tau) * target_param.data)
            # # update target actor 
            # for target_param, param in zip(env.leader_agent.target_actor.parameters(), env.leader_agent.actor.parameters()):
            #     target_param.data.copy_(env.leader_agent.tau * param.data + (1.0 - env.leader_agent.tau) * target_param.data)
        # ===============================================================
            

        multi_obs = multi_next_obs
        leader_episode_reward = multi_rewards["leader_agent"]

        if step_i % 100 == 0:
            print("episode",episode_i,"step",step_i, "      REWARD : ",leader_episode_reward, "    EPISODE_REWARD : ", episode_reward)

        if multi_done["leader_agent"]:
            break

        if infos["leader_agent"]:
            print("**************REACH GOAL****************")
            break

        # 4. save the agents' models
        
        # if episode_i == 0:
        #     leader_highest_reward = leader_episode_reward
        if leader_episode_reward > leader_highest_reward:
            leader_highest_reward = leader_episode_reward
            print(f"Highest leader reward updated at episode{episode_i}: {round(leader_highest_reward ,2)}")

            
            # flag = os.path.exists(agent_path)
            # if not flag:
            #     os.makedirs(agent_path)
            # # torch.save(env.leader_agent.actor.state_dict(), f"{agent_path}" + f"leader_agent_actor_{scenario}.pth")
            # env.leader_agent.actor.save_checkpoint(f"{agent_path}" + f"leader_agent_actor_{scenario}.pth")
            # env.leader_agent.target_actor.save_checkpoint(f"{agent_path}" + f"leader_agent_target_actor_{scenario}.pth")
            # env.leader_agent.critic.save_checkpoint(f"{agent_path}" + f"leader_agent_critic_{scenario}.pth")
            # env.leader_agent.target_critic.save_checkpoint(f"{agent_path}" + f"leader_agent_target_critic_{scenario}.pth")

        
        

        




    # 3. render the env
    if (episode_i + 1) % RENDER_FREQUENCE == 0:
        env = CustomEnv(delta=0.5)
        env.leader_agent.actor.load_checkpoint(f"{agent_path}" + f"leader_agent_actor_{scenario}.pth")
        for test_epi_i in range(1):
            multi_obs, infos = env.reset()
            print("rendering episode ", test_epi_i," ==========================")
            for step_i in range(RENDER_NUM_STEP):
                env.render(display_time=0.1)

                # multi_actions = {}
                # for agent_i, agent_name in enumerate(agent_name_list):
                #     agent = agents[agent_i]
                #     single_obs = multi_obs[agent_name]
                #     single_action = agent.get_action(single_obs)
                #     multi_actions[agent_name] = single_action
                MODE_render = "d"
                leader_action = env.leader_agent.get_action(multi_obs["leader_agent"], MODE_render)
                print("leader_action : ", leader_action)
                multi_actions = {}#follower_action
                for agent_id, agent in env.follower_agents.items():
                    
                    multi_actions[agent_id] = [0,0]
                # multi_next_obs, multi_rewards, multi_done, infos = env.step(multi_actions)
                multi_next_obs, multi_rewards, multi_done, infos = env.step(leader_action=leader_action, follower_actions=multi_actions)
                    
                multi_obs = multi_next_obs
            env.render_close()
    
        # # 4. save the agents' models
        # if episode_i == 0:
            
        #     leader_highest_reward = leader_episode_reward
    
        # if leader_episode_reward > leader_highest_reward:
        #     leader_highest_reward = leader_episode_reward
        #     print(f"Highest leader reward updated at episode{episode_i}: {round(leader_highest_reward / 512 ,2)}")

            
        #     flag = os.path.exists(agent_path)
        #     if not flag:
        #         os.makedirs(agent_path)
        #     # torch.save(env.leader_agent.actor.state_dict(), f"{agent_path}" + f"leader_agent_actor_{scenario}.pth")
        #     env.leader_agent.actor.save_checkpoint(f"{agent_path}" + f"leader_agent_actor_{scenario}.pth")
        


