import math

class obstacle():
    def __init__(self, radius, pos_x, pos_y, safe_theta) -> None:
        self.radius = radius
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.safe_theta = safe_theta

    def is_collision(self, agent_x, agent_y, agent_radius):
        dis = math.sqrt((agent_x - self.pos_x)**2 + (agent_y - self.pos_y)**2)
        safe_dis = self.safe_theta + agent_radius + self.radius
        if(dis < safe_dis):
            return False
        else:
            return True
        
    # dis and theta from obstacle to agent
    def calculate_dis_theta(self, agent_x, agent_y):
        dis = math.sqrt((agent_x - self.pos_x)**2 + (agent_y - self.pos_y)**2)
        
        theta = math.atan2(agent_y - self.pos_y, agent_x - self.pos_x)

        return dis,theta