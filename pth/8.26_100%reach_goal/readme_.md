计算vo的时候把障碍物的半径*1.2   
测试1000次结果 测试结果为：到达目标 907 次， 碰撞边缘 1 次，碰撞障碍 56 次，未到达目标 36 次      



def _caculate_leader_vo_reward(self, agent, target, agent_action):
        vx = agent.vel * cos(agent.orientation)
        vy = agent.vel * sin(agent.orientation)
        robot_state = [agent.pos[0], agent.pos[1], vx, vy, self.agent_radius]
        nei_state_list = []
        obs_cir_list = []
        for obs in self.obstacles.values():
            dis = np.linalg.norm(obs.position() - self.leader_agent.position())
            if dis <= self.obs_delta:
                obs_state = [obs.pos_x, obs.pos_y, obs.xy_vel[0], obs.xy_vel[1], self.obs_radius * 1.2]#放大
                obs_cir_list.append(obs_state)
        # obs_line_list = [ np.array([[0, 0], [0, 100]]) ,
        #                     np.array([[0, 100], [100, 100]]) ,
        #                     np.array([[100, 100], [100, 0]]) , 
        #                     np.array([[100, 0], [0, 0]]) ]
        obs_line_list = []
        action = [vx, vy]
        vo_flag, min_exp_time, min_dis = self.rvo_inter.config_vo_reward(robot_state=robot_state,
                                                                                                                                        nei_state_list=nei_state_list,
                                                                                                                                        obs_cir_list=obs_cir_list,
                                                                                                                                        obs_line_list=obs_line_list,
                                                                                                                                        action=action)
        if vo_flag :
            # reward = - 10/(min_exp_time + 0.1)
            reward = - 2
        else:
            reward = 0
        return vo_flag, reward *2, min_dis


def _caculate_target_reward(self, check_agent_id, check_agent, formation_target, action, t, last_distance, vo_flag):
        """和编队目标之间的距离"""
        reward = 0
        dis, angle = CustomEnv.calculate_relative_distance_and_angle(check_agent.pos, formation_target)

        # 设置最小阈值
        min_dis_threshold = self.target_radius/2
        if dis < min_dis_threshold:
            dis = min_dis_threshold

        
        if check_agent.done and check_agent.target:
            return 500
        
        else :
            dis_ = -(dis - last_distance) 

            if dis > self.width * 0.3 :
                dis_reward = 0
            # elif self.width * 0.2< dis <=self.width * 0.8:
            #     dis_reward = 1/dis
            elif dis <= self.width * 0.3:
                # dis_reward = 1.125- dis/self.width * 0.2
                dis_reward = min_dis_threshold/(dis + min_dis_threshold)
            
            reward = dis_
            # reward = dis_ + dis_reward/2
            # print(((action[0] + 1) /2)/1.5, - abs(action[1]) /2,dis_  ,dis_reward )
            # print((1- (dis_ / 100))*2)
            if vo_flag:
                # return 0
                return reward *120

            else:
                return reward *500
        # if np.isnan(reward) or np.isinf(reward):
        #     print(f"NaN or Inf detected in reward calculation! reward: {reward}, dis: {dis}, action: {action}")
        #     reward = -100  # 或其他合理的默认值

            

    def _caculate_obstacle_reward(self, agent_id, agent, last_obs_distance):
        if agent.done and not agent.target:
                    return -500
        
        reward = 0
        
        for obs_id, obs in self.obstacles.items():
            dis, angle = CustomEnv.calculate_relative_distance_and_angle(agent.pos, [obs.pos_x, obs.pos_y])
            if dis <= self.obs_delta and abs(angle - agent.orientation)<=np.pi/2:
                vx = agent.vel * cos(agent.orientation)
                vy = agent.vel * sin(agent.orientation)
                robot_state = [agent.pos[0], agent.pos[1], vx, vy, self.agent_radius]
                nei_state_list = []
                obs_cir_list = [ [obs.pos_x, obs.pos_y, obs.xy_vel[0], obs.xy_vel[1], self.obs_radius * 1.2]]#放大
                obs_line_list = []
                action = [vx, vy]
                vo_flag, min_exp_time, min_dis = self.rvo_inter.config_vo_reward(robot_state=robot_state,
                                                                                                                                                        nei_state_list=nei_state_list,
                                                                                                                                                        obs_cir_list=obs_cir_list,
                                                                                                                                                        obs_line_list=obs_line_list,
                                                                                                                                                        action=action)
                if vo_flag:
                    delta = 750
                    x = 60
                    # x = 0
                    # x = (dis - self.obs_delta)

                else:
                    delta = 240
                    x = 30

                d_dis = dis - last_obs_distance[obs_id]

                # if dis <= self.obs_delta :
                #     x = -(1/dis)
                # else:
                #     x = 0


                # if dis < 5:
                # x = -(1/dis)
                # else:
                #     x = 0
                
                        # print( f"d_dis : {d_dis}, x : {x}")
                    
                reward += d_dis * delta+ -(1/dis) * x
                # reward +=  x * 50
            
        return  reward 
    
    def _caculate_velocity_reward(self, agent, action, vo_flag):
        if vo_flag:
            return ((action[0] + 1) + abs(action[1])/2) *5
        else:
            return ((action[0] + 1) - abs(action[1])/2 ) *5
        

    def _caculate_side_reward(self, agent):
        reward = 0
        distances = [agent.pos[0], self.width - agent.pos[0], agent.pos[1], self.height - agent.pos[1]]
        
        for dis in distances:
            if dis > self.width * 0.18:
                re = 0
            
            elif self.width *0.05 < dis <= self.width* 0.18:
                re = -self.width *0.05 /dis
            elif dis <= self.width *0.05:
                if agent.done and not agent.target:
                    return -100
                re = -20 *((self.width *0.05)/(self.width *0.05 + dis))

            reward += re

        return reward *2






        def _caculate_obstacle_reward(self, agent_id, agent, last_obs_distance):
        if agent.done and not agent.target:
                    return -500
        
        reward = 0
        
        for obs_id, obs in self.obstacles.items():
            dis, angle = CustomEnv.calculate_relative_distance_and_angle(agent.pos, [obs.pos_x, obs.pos_y])
            if dis <= self.obs_delta and abs(angle - agent.orientation)<=np.pi/2:
                vx = agent.vel * cos(agent.orientation)
                vy = agent.vel * sin(agent.orientation)
                robot_state = [agent.pos[0], agent.pos[1], vx, vy, self.agent_radius]
                nei_state_list = []
                obs_cir_list = [ [obs.pos_x, obs.pos_y, obs.xy_vel[0], obs.xy_vel[1], self.obs_radius * 1.2]]#放大
                obs_line_list = []
                action = [vx, vy]
                vo_flag, min_exp_time, min_dis = self.rvo_inter.config_vo_reward(robot_state=robot_state,
                                                                                                                                                        nei_state_list=nei_state_list,
                                                                                                                                                        obs_cir_list=obs_cir_list,
                                                                                                                                                        obs_line_list=obs_line_list,
                                                                                                                                                        action=action)
                if vo_flag:
                    delta = 750
                    x = 60
                    # x = 0
                    # x = (dis - self.obs_delta)
                    relative_pos = obs.position() - agent.position()
                    # 计算相对位置的单位向量
                    relative_pos_unit = relative_pos / np.linalg.norm(relative_pos)
                    
                    angle =self.calculate_angle_between_vectors([vx, vy], relative_pos)

                    

                    



                else:
                    delta = 240
                    x = 30
                    angle = 0

                d_dis = dis - last_obs_distance[obs_id]

                # if dis <= self.obs_delta :
                #     x = -(1/dis)
                # else:
                #     x = 0


                # if dis < 5:
                # x = -(1/dis)
                # else:
                #     x = 0
                
                        # print( f"d_dis : {d_dis}, x : {x}")
                    
                # reward += d_dis * delta+ -(1/dis) * x
                reward += d_dis * delta +abs(angle)*100
                # reward +=  x * 50
            
        return  reward 