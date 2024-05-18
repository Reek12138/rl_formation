import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.env_base import CustomEnv

env = CustomEnv(delta=0.1)
env.render(display_time=1) 
# env.render_close()
# env.leader_agent.set_position(50,50)
# for i,agent in enumerate(env.follower_agents.values()):
#     agent.set_position(env.leader_agent.pos_x + (i+1)*10 + np.random.rand() * 5,
#                        env.leader_agent.pos_y + (i+1)*10 + np.random.rand() * 5)
# env.render(display_time=3)
# env.render_close()
for i in range(100):
    observations, rewards, dones, infos = env.step(leader_action=[10,(30/180)*np.pi], follower_actions={
                                                                                                                                                "agent_0":[10,10],
                                                                                                                                                "agent_1":[10,20],
                                                                                                                                                "agent_2":[10,5],
                                                                                                                                                "agent_3":[10,10]
    })
    # 每10轮打印一次结果
    if (i + 1) % 10 == 0:
        print("---------------------------------------------------------------------------------------------------------")
        print(f"Round {i+1}: Observations: {observations}")
        print(f"Round {i+1}: Rewards: {rewards}")
        print(f"Round {i+1}: Dones: {dones}")
        print(f"Round {i+1}: Infos: {infos}")
        # env.reset()
    env.render(display_time=0.1)
env.render_close()
