import numpy as np
import math
from physics_sim import PhysicsSim

class Land_Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
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
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        #
        # Aim to have the quadcopter land flat on the ground plane, i.e.  elevation Z = 0
        #
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 0.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # Include a factor to ensure there is no exagerated tilting
        # Penalise for large angular movement, example theta:
        # 1 - sin(theta), = 1 if theta = 0 and = 0 if theta -> 90 degrees
        # Introduce factor (1-sin(theta) * (1 - sin(phi)) * (1 - sin(psi))

        #
        # Keep reward in the vicinity of 0 - 1, introduce another penalty
        # exp(-1/r) where r is the pythagorean distance from current 
        # position to target.

        #
        # exp(-1/r) -> 0 as r -> 0
        # exp(-1/r) -> 1 as r -> inf

        penalty = (1. - abs(math.sin(self.sim.pose[3])))
        penalty *= (1. - abs(math.sin(self.sim.pose[4])))
        penalty *= (1. - abs(math.sin(self.sim.pose[5])))

        delta = abs(self.sim.pose[:3] - self.target_pos)
        r = math.sqrt(np.dot(delta, delta))
        
        if(r > 0.01): decay = math.exp(-1/r) # Give range -1 to 1
        else: decay = 0
        reward = 1. - decay
        reward *= penalty
        return reward

    def getPose(self):
        return self.sim.pose

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
