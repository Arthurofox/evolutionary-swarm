import torch
import torch.nn as nn
from pufferlib.models import LSTMWrapper

class SwarmPolicy(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # Input size is the flattened observation (5x5 = 25)
        self.obs_size = envs.single_observation_space.shape[0]
        
        # Simple Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Decoders (Heads) for actions
        # Head 1: Movement (5 options)
        self.actor_move = nn.Linear(64, 5)
        # Head 2: Signal (3 options)
        self.actor_signal = nn.Linear(64, 3)
        
        # Critic (Value function)
        self.critic = nn.Linear(64, 1)

    def get_value(self, x, state=None):
        return self.critic(self.encoder(x))

    def get_action_and_value(self, x, state=None, action=None):
        hidden = self.encoder(x)
        
        # Calculate logits for both heads
        logits_move = self.actor_move(hidden)
        logits_signal = self.actor_signal(hidden)
        
        # Create distributions
        probs_move = torch.distributions.Categorical(logits=logits_move)
        probs_signal = torch.distributions.Categorical(logits=logits_signal)
        
        if action is None:
            move = probs_move.sample()
            signal = probs_signal.sample()
        else:
            # PufferLib might pass action as [batch, 2]
            move = action[:, 0]
            signal = action[:, 1]
            
        # Calculate log probs (sum them for independent actions)
        log_prob = probs_move.log_prob(move) + probs_signal.log_prob(signal)
        entropy = probs_move.entropy() + probs_signal.entropy()
        
        # Stack actions to return [batch, 2]
        return torch.stack([move, signal], dim=1), log_prob, entropy, self.critic(hidden), state