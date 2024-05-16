import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.env_base import CustomEnv

env = CustomEnv()
env.render(display_time=5)