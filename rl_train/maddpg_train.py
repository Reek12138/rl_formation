import gym
# import test_env
# from pettingzoo.mpe import simple_adversary_v3
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device:{device}")

def multi_obs_to_state(multi_obs):
    state = np.array([])
    for agent_id, agent_obs in multi_obs.items():
        if agent_id != "leader_agent":
            state = np.concatenate([state, agent_obs])
    return state

NUM_EPISODE = 10000
NUM_STEP = 200
MEMORY_SIZE = 100000
BATCH_SIZE = 512
TARGET_UPDATE_INTERVAL=200
LR_ACTOR = 0.01
LR_CRITIC = 0.01
HIDDEN_DIM = 64
GAMMA = 0.99
TAU = 0.01

#1. initialize the agent
# env = simple_adversary_v3.parallel_env(N= 2, max_cycles = NUM_STEP, continuous_actions = True)
env = CustomEnv(delta=0.1)
# print("reset1==========================")

multi_obs, infos = env.reset()
#这里要根据环境修改
NUM_AGENT = env.num_agents
agent_name_list = env.follower_agents.keys()


scenario = "my_env"
current_path = os.path.dirname(os.path.realpath(__file__))
agent_path = current_path + "/models/" + scenario + '/'
timestamp = time.strftime("%Y%m%d%H%M%S")

# 1.1 get obs_dim
#这里只拿follower的observation_dim
obs_dim = []
for agent_id,agent_obs in multi_obs.items():
    if agent_id != "leader_agent":
        obs_dim.append(agent_obs.shape[0])
state_dim = sum(obs_dim)

# 1.2 get action_dim 
action_dim = []
for agent_name in agent_name_list:
    if agent_name != "leader_agent":
    # action_dim.append(env.action_space(agent_name).sample().shape[0])
        action_dim.append(2)
    

agents = [ ]
for agent in env.follower_agents.values():
    agents.append(agent)

# 2. main training loop
for episode_i in range(NUM_EPISODE):
    # print("reset2==========================")
    print("episode",episode_i,"========================")

    multi_obs, infos = env.reset()

    episode_reward = 0
    multi_done = {agent_name: False for agent_name in agent_name_list}

    for step_i in range(NUM_STEP):
        if step_i % 50 == 0:
            print("episode",episode_i,"step",step_i)

        total_step = episode_i*NUM_STEP + step_i

        #2.1 collect actions from all agents 
        leader_action = env.leader_agent.get_action(multi_obs["leader_agent"])

        multi_actions = {}#follower_action
        for agent_id, agent in env.follower_agents.items():
            single_obs = multi_obs[agent_id]
            single_action = agent.get_action(single_obs)
            multi_actions[agent_id] = single_action

        # 2.2 excute action 
        multi_next_obs, multi_rewards, multi_done, infos = env.step(leader_action=leader_action, follower_actions=multi_actions)
        follower_state = multi_obs_to_state(multi_obs)#把观测量转化为全局观测量,follower不包含对leader位置的观测
        follower_next_state = multi_obs_to_state(multi_next_obs)

        if step_i >= NUM_STEP -1:
            # multi_done = {agnet_name :True for agnet_name in agent_name_list}
            env.leader_agent.done = True
            multi_done["leader_agent"] = env.leader_agent.done
            for agent_id, agent in env.follower_agents.items():
                agent.done = True
                multi_done[agent_id] = agent.done
            
        # 2.3 story memory 每个agent有自己的buffer
        env.leader_agent.replay_buffer.add_memo(multi_obs["leader_agent"],
                                                multi_next_obs["leader_agent"],
                                                np.array(multi_obs["leader_agent"]),
                                                np.array(multi_next_obs["leader_agent"]),
                                                leader_action,
                                                multi_rewards["leader_agent"],
                                                multi_done["leader_agent"]
                                                )
        # print("add_leader_memo")
        for agent_id, agent in env.follower_agents.items():
            single_obs = multi_obs[agent_id]
            single_next_obs = multi_next_obs[agent_id]
            single_action = multi_actions[agent_id]
            single_reward = multi_rewards[agent_id]
            single_done = multi_done[agent_id]
            # print(agent_id, single_obs)
            agent.replay_buffer.add_memo(single_obs, 
                                        single_next_obs,
                                        follower_state, 
                                        follower_next_state, 
                                        single_action, 
                                        single_reward, 
                                        single_done)



        # 2.4  update brain every fixed steps
        multi_batch_obses = []
        multi_batch_next_obs = []
        multi_batch_states= []
        multi_batch_next_states= []
        multi_batch_actions = []
        multi_batch_next_actions = []
        multi_batch_online_actions = []
        multi_batch_rewards = []
        multi_batch_dones = []

        # 2.4.1 sample a batch of memories
        current_memo_size = min(MEMORY_SIZE, total_step+1)
        if current_memo_size < BATCH_SIZE:
            batch_idx = range(0, current_memo_size)
        else:
            batch_idx = np.random.choice(current_memo_size,  BATCH_SIZE, replace=False)        

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

        #target actor output
        leader_single_batch_next_actions = env.leader_agent.target_actor.forward(leader_batch_next_obses_tensor)

        #actor output
        leader_single_batch_online_action = env.leader_agent.actor.forward(leader_batch_obses_tensor)

        #follower agent update
        for agend_id, agent in env.follower_agents.items():
            batch_obses, batch_next_obses, batch_states, batch_next_state, batch_actions, batch_rewards, batch_dones  = agent.replay_buffer.sample(batch_idx)
            # single+batch 
            batch_obses_tensor = torch.tensor(batch_obses, dtype = torch.float).to(device)
            batch_next_obses_tensor = torch.tensor(batch_next_obses, dtype = torch.float).to(device)
            batch_states_tensor = torch.tensor(batch_states, dtype = torch.float).to(device)
            batch_next_state_tensor = torch.tensor(batch_next_state, dtype = torch.float).to(device)
            batch_actions_tensor = torch.tensor(batch_actions, dtype = torch.float).to(device)
            batch_rewards_tensor = torch.tensor(batch_rewards, dtype = torch.float).to(device)
            batch_dones_tensor = torch.tensor(batch_dones, dtype = torch.float).to(device)
            # multiple+batch 
            multi_batch_obses.append(batch_obses_tensor)
            multi_batch_next_obs.append(batch_next_obses_tensor)
            multi_batch_states.append(batch_states_tensor)
            multi_batch_next_states.append(batch_next_state_tensor)
            multi_batch_actions.append(batch_actions_tensor)
            
            #target actor output
            # single_batch_next_actions = agent.target_actor.forward(multi_batch_next_obs)
            single_batch_next_actions = agent.target_actor.forward(batch_next_obses_tensor)
            multi_batch_next_actions.append(single_batch_next_actions)
            # actor output
            # single_batch_online_action = agent.actor.forward(multi_batch_obses)
            single_batch_online_action = agent.actor.forward(batch_obses_tensor)
            multi_batch_online_actions.append(single_batch_online_action)

            multi_batch_rewards.append(batch_rewards_tensor)
            multi_batch_dones.append(batch_dones_tensor)

        # 采样的action
        multi_batch_actions_tensor = torch.cat(multi_batch_actions, dim =1).to(device)
        # 采样的next_action
        multi_batch_next_actions_tensor = torch.cat(multi_batch_next_actions, dim =1).to(device)
        # 采样的state经过网络对应的action
        multi_batch_online_actions_tensor = torch.cat(multi_batch_online_actions, dim =1).to(device)

        # update critic and actor 
        if(total_step +1)% TARGET_UPDATE_INTERVAL == 0:
            # for agent_i in range(NUM_AGENT):
            leader_critic_target_q = env.leader_agent.target_critic.forward(leader_batch_next_state_tensor, 
                                                                            leader_batch_actions_tensor.detach())
            leader_y = (leader_batch_rewards_tensor + (1 - leader_batch_dones_tensor) * env.leader_agent.gamma * leader_critic_target_q).flatten()

            leader_critic_q = env.leader_agent.critic.forward(leader_batch_states_tensor, leader_batch_actions_tensor.detach()).flatten()

            leader_critic_loss = nn.MSELoss()(leader_y, leader_critic_q)
            env.leader_agent.critic.optimizer.zero_grad()
            leader_critic_loss.backward()
            env.leader_agent.critic.optimizer.step()

            leader_actor_loss = env.leader_agent.critic.forward(leader_batch_states_tensor,
                                                                leader_single_batch_online_action.detach()).flatten()
            leader_actor_loss = -torch.mean(leader_actor_loss)
            env.leader_agent.actor.optimizer.zero_grad()
            leader_actor_loss.backward()
            env.leader_agent.actor.optimizer.step()

            # update target critic 
            for target_param, param in zip(env.leader_agent.target_critic.parameters(), env.leader_agent.critic.parameters()):
                target_param.data.copy_(env.leader_agent.tau * param.data + (1.0 - env.leader_agent.tau) * target_param.data)
            # update target actor 
            for target_param, param in zip(env.leader_agent.target_actor.parameters(), env.leader_agent.actor.parameters()):
                target_param.data.copy_(env.leader_agent.tau * param.data + (1.0 - env.leader_agent.tau) * target_param.data)

            for agent_i, agent in enumerate(env.follower_agents.values()):
                # agent = agents[agent_i]

                batch_obses_tensor = multi_batch_obses[agent_i]
                batch_states_tensor = multi_batch_states[agent_i]
                batch_next_state_tensor = multi_batch_next_states[agent_i]
                batch_rewards_tensor = multi_batch_rewards[agent_i]
                batch_dones_tensor = multi_batch_dones[agent_i]
                batch_actions_tensor = multi_batch_actions[agent_i]

                # target critic
                critic_target_q = agent.target_critic.forward(batch_next_state_tensor,
                                                              multi_batch_actions_tensor.detach())
                y = (batch_rewards_tensor + (1 - batch_dones_tensor) * agent.gamma * critic_target_q).flatten()

                critic_q = agent.critic.forward(batch_states_tensor, multi_batch_actions_tensor.detach()).flatten()

                # critic update
                critic_loss = nn.MSELoss()(y, critic_q)
                agent.critic.optimizer.zero_grad()
                critic_loss.backward()
                # critic_loss.backward(retain_graph=True) 
                agent.critic.optimizer.step()

                # actor update 
                actor_loss = agent.critic.forward(batch_states_tensor,
                                                  multi_batch_online_actions_tensor.detach()).flatten()
                actor_loss = -torch.mean(actor_loss)
                agent.actor.optimizer.zero_grad()
                actor_loss.backward()
                agent.actor.optimizer.step()

                # update target critic 
                for target_param, param in zip(agent.target_critic.parameters(), agent.critic.parameters()):
                    target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)
                # update target actor 
                for target_param, param in zip(agent.target_actor.parameters(), agent.actor.parameters()):
                    target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)
                

        multi_obs = multi_next_obs
        episode_reward += sum(single_reward for single_reward in multi_rewards.values())
        # print(f"episode_reward:{episode_reward}")
        
    # 3. render the env
    if (episode_i + 1) % 100 == 0:
        env = CustomEnv()
        for test_epi_i in range(1):
            multi_obs, infos = env.reset()
            print("rendering episode ", test_epi_i," ==========================")
            for step_i in range(NUM_STEP):
                env.render(display_time=0.1)

                # multi_actions = {}
                # for agent_i, agent_name in enumerate(agent_name_list):
                #     agent = agents[agent_i]
                #     single_obs = multi_obs[agent_name]
                #     single_action = agent.get_action(single_obs)
                #     multi_actions[agent_name] = single_action
                leader_action = env.leader_agent.get_action(multi_obs["leader_agent"])

                multi_actions = {}#follower_action
                for agent_id, agent in env.follower_agents.items():
                    single_obs = multi_obs[agent_id]
                    single_action = agent.get_action(single_obs)
                    multi_actions[agent_id] = single_action
                # multi_next_obs, multi_rewards, multi_done, infos = env.step(multi_actions)
                multi_next_obs, multi_rewards, multi_done, infos = env.step(leader_action=leader_action, follower_actions=multi_actions)
                    
                multi_obs = multi_next_obs
                env.render_close()

    # 4. save the agents' models
    if episode_i == 0:
        highest_reward = episode_reward
    if episode_reward > highest_reward:
        highest_reward = episode_reward
        print(f"Highest reward updated at episode{episode_i}: {round(highest_reward,2)}")
        
        flag = os.path.exists(agent_path)
        if not flag:
            os.makedirs(agent_path)
        torch.save(env.leader_agent.actor.state_dict(), f"{agent_path}" + f"leader_agent_actor_{scenario}.pth")

        for agent in env.follower_agents.values():
            # if not flag:
            #     os.makedirs(agent_path)
            # torch.save(agent.actor.state_dict(), f"{agent_path}" + f"agent_{agent_i}_actor_{scenario}_{timestamp}.pth")
            torch.save(agent.actor.state_dict(), f"{agent_path}" + f"agent_{agent_i}_actor_{scenario}.pth")



env.close()