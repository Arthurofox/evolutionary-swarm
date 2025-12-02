# train.py
import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import pufferlib.vector
import pufferlib.emulation

from env import BlindSwarm
from model import SwarmPolicy

# --- CONFIGURATION ---
TOTAL_TIMESTEPS = 2_000_000
LEARNING_RATE = 2.5e-4
NUM_ENVS = 4                 
NUM_AGENTS = 64              
NUM_STEPS = 128              
VISUALIZE_EVERY = 50         

# --- FIX: Use PettingZoo Wrapper ---
def make_env(*args, **kwargs):
    env = BlindSwarm(num_agents=NUM_AGENTS, grid_size=32)
    # This wrapper handles the conversion from Dictionary -> Batch for us
    return pufferlib.emulation.PettingZooPufferEnv(env=env)

def watch_evolution(agent, device):
    print("\n🍿 Watching evolution...")
    env = BlindSwarm(num_agents=NUM_AGENTS, grid_size=32, render_mode="human")
    
    # PettingZoo reset returns (obs, info)
    obs_dict, _ = env.reset()
    
    for _ in range(200):
        # Convert Dict -> List -> Tensor
        obs_list = [obs_dict[agent] for agent in env.agents]
        obs_tensor = torch.Tensor(np.array(obs_list)).to(device)
        
        with torch.no_grad():
            actions, _, _, _, _ = agent.get_action_and_value(obs_tensor)
        
        # Convert Tensor -> List -> Dict
        actions_np = actions.cpu().numpy()
        action_dict = {agent: actions_np[i] for i, agent in enumerate(env.agents)}
        
        obs_dict, _, _, _, _ = env.step(action_dict)
        
        import pygame
        pygame.time.wait(30)
        
    env.close()
    print("✅ Resuming training...\n")

def train():
    run_name = f"Swarm_{int(time.time())}"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Training on {device}")

    vec_env = pufferlib.vector.make(
        make_env,
        backend=pufferlib.vector.Serial, 
        num_envs=NUM_ENVS
    )

    agent = SwarmPolicy(vec_env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    obs = torch.zeros((NUM_STEPS, NUM_ENVS * NUM_AGENTS, agent.obs_size)).to(device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS * NUM_AGENTS, 2)).to(device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS * NUM_AGENTS)).to(device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS * NUM_AGENTS)).to(device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS * NUM_AGENTS)).to(device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS * NUM_AGENTS)).to(device)

    global_step = 0
    next_obs, _ = vec_env.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(NUM_ENVS * NUM_AGENTS).to(device)

    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * NUM_AGENTS * NUM_STEPS)
    
    for update in range(1, num_updates + 1):
        
        if update % VISUALIZE_EVERY == 0:
            watch_evolution(agent, device)

        # 1. Collect Data
        for step in range(NUM_STEPS):
            global_step += NUM_ENVS * NUM_AGENTS
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, term, trunc, info = vec_env.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.logical_or(torch.tensor(term), torch.tensor(trunc)).to(device).float()

        # 2. Bootstrap
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + 0.99 * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
            returns = advantages + values

        # 3. Train
        b_obs = obs.reshape((-1, agent.obs_size))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, 2))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(len(b_obs))
        for epoch in range(4):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_obs), 32):
                end = start + 32
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], action=b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                pg_loss = -torch.min(
                    mb_advantages * ratio,
                    mb_advantages * torch.clamp(ratio, 0.8, 1.2)
                ).mean()
                
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                loss = pg_loss - (0.01 * entropy.mean()) + (0.5 * v_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Step {global_step} | Avg Reward: {rewards.mean().item():.4f}")

    vec_env.close()

if __name__ == "__main__":
    train()