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
for _ in range(1000):
    observations, rewards, dones, infos = env.step(leader_action=[10,0], follower_actions={
                                                                                                                                                "agent_0":[10,0],
                                                                                                                                                "agent_1":[10,0],
                                                                                                                                                "agent_2":[10,0],
                                                                                                                                                "agent_3":[10,0]
    })
    env.render(display_time=0.1)
env.render_close()
