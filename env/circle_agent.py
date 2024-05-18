import numpy as np
class circle_agent():
    def __init__(self, radius, pos_x, pos_y, linear_vel = 0, orientation_vel = 0, orientation = 0, vel_x=0, vel_y=0) -> None:
        self.radius = radius
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.observation = {}
        self.reward = 0.0
        self.done = False
        self.info = {}
        self.xy_vel = [vel_x, vel_y]
        self.linear_orientation = [linear_vel,  orientation_vel]
        self.orientation = orientation

        
    
    def set_position(self, x, y):
        """设置代理的当前位置"""
        self.pos_x = x
        self.pos_y = y
    def position(self):
        return np.array([self.pos_x, self.pos_y])
    
    def set_xy_vel(self, vx, vy):
        self.vel = [vx, vy]

    def set_linear_orientation(self, linear_vel, orientation):
        self.linear_orientation = [linear_vel, orientation]