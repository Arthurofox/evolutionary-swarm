import numpy as np
import pygame
import functools
from gymnasium.spaces import Box, MultiDiscrete
from pettingzoo import ParallelEnv

# --- FIX: Callable Spaces for PufferLib Compatibility ---
class CallableBox(Box):
    def __call__(self, agent=None):
        return self

class CallableMultiDiscrete(MultiDiscrete):
    def __call__(self, agent=None):
        return self
# --------------------------------------------------------

class BlindSwarm(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "blind_swarm_v1"}

    def __init__(self, num_agents=64, grid_size=32, render_mode=None):
        # FIX: Renamed variable to avoid conflict with ParallelEnv.num_agents property
        self.n_agents = num_agents 
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.screen = None
        self.cell_size = 12
        
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]

        # Define spaces using Callable wrappers
        self.obs_size = 25 
        # self.observation_space and self.action_space are now callable objects
        # that also satisfy isinstance(obj, Box/MultiDiscrete)
        self.observation_space = CallableBox(low=0, high=4, shape=(self.obs_size,), dtype=np.float32)
        self.action_space = CallableMultiDiscrete([5, 3])

    # Methods removed as they are replaced by callable instance attributes
    # def observation_space(self, agent):
    #     return self._single_observation_space
    #
    # def action_space(self, agent):
    #     return self._single_action_space

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.grid_static = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Spawn food
        food_locs = np.random.randint(0, self.grid_size, (40, 2))
        for f in food_locs:
            self.grid_static[f[0], f[1]] = 1
            
        # FIX: Use self.n_agents here
        self.agent_positions = np.random.randint(0, self.grid_size, (self.n_agents, 2))
        self.agent_signals = np.zeros(self.n_agents, dtype=int)
        self.step_count = 0
        
        obs_array = self._get_obs_array()
        observations = {agent: obs_array[i] for i, agent in enumerate(self.agents)}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def _get_obs_array(self):
        signal_map = np.zeros((self.grid_size, self.grid_size), dtype=int)
        active = self.agent_signals > 0
        if np.any(active):
            signal_map[self.agent_positions[active, 0], self.agent_positions[active, 1]] = self.agent_signals[active] + 2

        global_view = np.maximum(self.grid_static, signal_map)
        
        pad = 2
        padded_view = np.pad(global_view, pad, mode='constant', constant_values=0)
        shifted_agents = self.agent_positions + pad
        
        obs_batch = []
        # FIX: Use self.n_agents here
        for i in range(self.n_agents):
            r, c = shifted_agents[i]
            window = padded_view[r-pad:r+pad+1, c-pad:c+pad+1]
            obs_batch.append(window.flatten())
            
        return np.array(obs_batch, dtype=np.float32)

    def step(self, actions):
        move_cmds = []
        sig_cmds = []
        
        for agent in self.agents:
            act = actions[agent]
            move_cmds.append(act[0])
            sig_cmds.append(act[1])
            
        moves = np.array(move_cmds)
        signals = np.array(sig_cmds)
        
        self.agent_signals = signals
        self.step_count += 1
        
        deltas = np.array([[0,0], [-1,0], [1,0], [0,-1], [0,1]])
        potential_pos = self.agent_positions + deltas[moves]
        self.agent_positions = np.clip(potential_pos, 0, self.grid_size - 1)
        
        # FIX: Use self.n_agents here
        rewards_arr = np.zeros(self.n_agents, dtype=np.float32)
        
        is_food = self.grid_static[self.agent_positions[:, 0], self.agent_positions[:, 1]] == 1
        rewards_arr[is_food] = 1.0
        self.grid_static[self.agent_positions[is_food, 0], self.agent_positions[is_food, 1]] = 0
        
        if np.any(is_food):
            num_eaten = np.sum(is_food)
            new_food = np.random.randint(0, self.grid_size, (num_eaten, 2))
            for f in new_food:
                self.grid_static[f[0], f[1]] = 1
        
        rewards_arr -= 0.01
        
        truncated = self.step_count >= 128
        obs_array = self._get_obs_array()
        
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for i, agent in enumerate(self.agents):
            observations[agent] = obs_array[i]
            rewards[agent] = rewards_arr[i]
            terminations[agent] = False
            truncations[agent] = truncated
            infos[agent] = {}

        if self.render_mode == "human":
            self.render()
            
        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size*self.cell_size, self.grid_size*self.cell_size))
        
        self.screen.fill((20, 20, 20))
        rows, cols = np.where(self.grid_static == 1)
        for r, c in zip(rows, cols):
            pygame.draw.rect(self.screen, (0, 200, 0), (c*self.cell_size, r*self.cell_size, self.cell_size, self.cell_size))
            
        for i, (r, c) in enumerate(self.agent_positions):
            color = (200, 0, 0)
            if self.agent_signals[i] == 1: color = (50, 50, 255)
            if self.agent_signals[i] == 2: color = (255, 255, 0)
            pygame.draw.rect(self.screen, color, (c*self.cell_size, r*self.cell_size, self.cell_size, self.cell_size))
        
        pygame.display.flip()
        pygame.event.pump()

    def close(self):
        if self.screen is not None:
            pygame.quit()