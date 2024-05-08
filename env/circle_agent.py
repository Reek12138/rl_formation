import numpy as np
class circle_agent():
    def __init__(self, radius, pos_x, pos_y) -> None:
        self.radius = radius
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.observation = {}
        self.reward = 0.0
        self.done = False
        self.info = {}
        
    
    def set_position(self, x, y):
        """设置代理的当前位置"""
        self.pos_x = x
        self.pos_y = y
    def position(self):
        return np.array([self.pos_x, self.pos_y])