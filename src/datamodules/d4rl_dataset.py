import gym
import d4rl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging

class D4RLTrajectoryDataset(Dataset):
    def __init__(self, env_name, horizon=100, rtg_scale=1000.0):
        logging.info(f"Loading D4RL dataset: {env_name}")
        self.env = gym.make(env_name)
        self.dataset = d4rl.qlearning_dataset(self.env)
        self.horizon = horizon
        self.rtg_scale = rtg_scale
        
        self.obs = self.dataset['observations']
        self.actions = self.dataset['actions']
        self.rewards = self.dataset['rewards']
        self.terminals = self.dataset['terminals']
        
        # Z-Score Normalization
        self.stats = {
            'obs_mean': self.obs.mean(axis=0),
            'obs_std': self.obs.std(axis=0) + 1e-6,
            'act_mean': self.actions.mean(axis=0),
            'act_std': self.actions.std(axis=0) + 1e-6,
        }
        
        self.obs = (self.obs - self.stats['obs_mean']) / self.stats['obs_std']
        self.actions = (self.actions - self.stats['act_mean']) / self.stats['act_std']
        
        # Calculate and scale Returns-to-go
        self.returns_to_go = self._calculate_returns_to_go() / self.rtg_scale
        
        self.indices = np.where(self.terminals == 0)[0]
        self.indices = [i for i in self.indices if i + horizon <= len(self.obs)]
        logging.info(f"Dataset loaded. Sequences: {len(self.indices)} | RTG Scale: {rtg_scale}")

    def _calculate_returns_to_go(self):
        returns = np.zeros_like(self.rewards)
        prev_return = 0
        for i in reversed(range(len(self.rewards))):
            if self.terminals[i]: prev_return = 0
            returns[i] = self.rewards[i] + prev_return
            prev_return = returns[i]
        return returns

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.horizon
        trajectories = np.concatenate([self.obs[start:end], self.actions[start:end]], axis=-1)
        return {
            "trajectories": torch.from_numpy(trajectories).float(),
            "returns": torch.from_numpy(self.returns_to_go[start:start+1]).float()
        }

    def unnormalize_action(self, action):
        return action * self.stats['act_std'] + self.stats['act_mean']

    def normalize_obs(self, obs):
        return (obs - self.stats['obs_mean']) / self.stats['obs_std']

def get_dataloader(env_name, horizon, batch_size, num_workers=4):
    dataset = D4RLTrajectoryDataset(env_name, horizon)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
