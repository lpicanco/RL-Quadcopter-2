import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None, task_type = 'take-off'):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.init_pose = init_pose if init_pose is not None else np.array([0., 0., 0., 0., 0., 0.]) 
        self.previous_pos = self.init_pose
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
        
        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        self.task_type = task_type

    def get_reward(self, done):
        """Uses current pose of sim to return reward."""
        distance_to_target = (abs(self.sim.pose[:3] - self.target_pos)).sum()
        if self.task_type == 'take-off':
            current_position = self.sim.pose[:3]
            distance_max = self.target_pos - self.init_pose[:3] + 1e-1
            mov_x = (self.target_pos[0] - current_position[0]) / (distance_max[0])
            mov_y = (self.target_pos[1] - current_position[1]) / (distance_max[1])
            mov_z = (self.target_pos[2] - current_position[2]) / (distance_max[2])

            reward = self.target_pos[2]*10 + 0.0001 * (-1 * mov_x**2  - 1 * mov_y**2 - 10 * mov_z**2)
            if self.previous_pos[2] > current_position[2] and self.sim.pose[2] < self.target_pos[2]:
                reward = -reward
 
            if distance_to_target <= 1.:
                reward += 10000
                done = True

            if self.sim.pose[2] > self.target_pos[2] + 3:
                reward = -1000

            if done and np.average(abs(distance_to_target)) > 6 and self.sim.time < self.sim.runtime:
                reward = reward * 0.3 # crash penalty

        elif self.task_type == "fly-to":
            distance_max = self.target_pos - self.init_pose[:3] + 1e-1
            distance_to_target = (self.target_pos - self.sim.pose[:3])
            diff_x = (distance_to_target[0]/distance_max[0])**2
            diff_y = (distance_to_target[1]/distance_max[1])**2
            diff_z = (distance_to_target[2]/distance_max[2])**2
            reward = (self.target_pos[2]*10 + 0.0001 * (-1 * diff_x - 1 * diff_y - 10 * diff_z))
            if self.previous_pos[2] > self.sim.pose[2] and self.sim.pose[2] < self.target_pos[2]:
                reward = -reward

            if done and np.average(abs(distance_to_target)) > 6:
                reward -= 10000 # crash penalty
            
            min_distance = 3
            if distance_to_target[0] <= min_distance and distance_to_target[1] <= min_distance and distance_to_target[2] <= min_distance:
                reward = reward*2
                done = True
        
        return reward, done

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        self.previous_pos = self.sim.pose
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward_step, done = self.get_reward(done) 
            reward += reward_step
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        self.previous_pos = self.init_pose
        return state