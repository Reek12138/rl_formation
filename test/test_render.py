import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.env_base import CustomEnv

env = CustomEnv()
env.render(display_time=3)
env.render_close()
# env.leader_agent.set_position(50,50)
# for i,agent in enumerate(env.follower_agents.values()):
#     agent.set_position(env.leader_agent.pos_x + (i+1)*10 + np.random.rand() * 5,
#                        env.leader_agent.pos_y + (i+1)*10 + np.random.rand() * 5)
# env.render(display_time=3)
# env.render_close()