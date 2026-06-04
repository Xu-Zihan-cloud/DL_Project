import gym
import d4rl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class D4RLTrajectoryDataset(Dataset):
    def __init__(self, env_name, horizon=100, terminal_penalty=0):
        self.env = gym.make(env_name)
        self.dataset = d4rl.qlearning_dataset(self.env)
        self.horizon = horizon
        
        self.obs = self.dataset['observations']
        self.actions = self.dataset['actions']
        self.next_obs = self.dataset['next_observations']
        self.rewards = self.dataset['rewards']
        self.terminals = self.dataset['terminals']
        
        # Calculate returns-to-go
        self.returns_to_go = self._calculate_returns_to_go()
        
        # Simple sequence extraction (can be optimized)
        self.indices = np.where(self.terminals == 0)[0]
        self.indices = [i for i in self.indices if i + horizon <= len(self.obs)]

    def _calculate_returns_to_go(self):
        # Implementation of return-to-go calculation
        returns = np.zeros_like(self.rewards)
        prev_return = 0
        for i in reversed(range(len(self.rewards))):
            if self.terminals[i]:
                prev_return = 0
            returns[i] = self.rewards[i] + prev_return
            prev_return = returns[i]
        return returns

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.horizon
        
        obs = self.obs[start:end]
        actions = self.actions[start:end]
        returns = self.returns_to_go[start:end]
        
        # Concatenate for cleandiffuser format [obs, action]
        trajectories = np.concatenate([obs, actions], axis=-1)
        
        return {
            "trajectories": torch.from_numpy(trajectories).float(),
            "returns": torch.from_numpy(returns[[0]]).float() # Use initial RTG as condition
        }

def get_dataloader(env_name, horizon, batch_size, num_workers=4):
    dataset = D4RLTrajectoryDataset(env_name, horizon)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
