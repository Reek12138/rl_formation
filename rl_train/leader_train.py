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

env = CustomEnv(delta=0.1)
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
for episode_i in range(NUM_EPISODE):
    # print("reset2==========================")
    print("episode",episode_i,"========================")

    multi_obs, infos = env.reset()
    leader_episode_reward = 0
    multi_done = {agent_name: False for agent_name in agent_name_list}

    for step_i in range(NUM_STEP):
        if step_i % 50 == 0:
            print("episode",episode_i,"step",step_i)

        total_step = episode_i*NUM_STEP + step_i

        #2.1 collect actions from all agents 
        leader_action = env.leader_agent.get_action(multi_obs["leader_agent"])

        multi_actions = {}#follower_action
        for agent_id, agent in env.follower_agents.items():
            multi_actions[agent_id]=[0,0]
            
        multi_next_obs, multi_rewards, multi_done, infos = env.step(leader_action=leader_action, follower_actions=multi_actions)

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

        multi_obs = multi_next_obs
        leader_reward = multi_rewards["leader_agent"]
    
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
                    
                    multi_actions[agent_id] = [0,0]
                # multi_next_obs, multi_rewards, multi_done, infos = env.step(multi_actions)
                multi_next_obs, multi_rewards, multi_done, infos = env.step(leader_action=leader_action, follower_actions=multi_actions)
                    
                multi_obs = multi_next_obs
            env.render_close()
    
    # 4. save the agents' models
    if episode_i == 0:
        
        leader_highest_reward = leader_episode_reward
   
    if leader_episode_reward > leader_highest_reward:
        leader_highest_reward = leader_episode_reward
        print(f"Highest leader reward updated at episode{episode_i}: {round(leader_highest_reward / 512 ,2)}")

        
        flag = os.path.exists(agent_path)
        if not flag:
            os.makedirs(agent_path)
        torch.save(env.leader_agent.actor.state_dict(), f"{agent_path}" + f"leader_agent_actor_{scenario}.pth")

       


