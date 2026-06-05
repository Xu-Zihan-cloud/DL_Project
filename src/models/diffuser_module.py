import torch
import torch.nn as nn
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_diffusion.mlps import MlpNNDiffusion

class DecisionDiffuserWrapper(nn.Module):
    def __init__(self, obs_dim, action_dim, horizon, model_config):
        super().__init__()
        self.obs_dim, self.action_dim, self.horizon = obs_dim, action_dim, horizon
        input_dim = horizon * (obs_dim + action_dim)
        
        # Use a standard MLP backbone
        self.nn_diffusion = MlpNNDiffusion(
            x_dim=input_dim, 
            hidden_dims=[512, 512, 512] # Increased capacity for full trajectory MLP
        )
        
        self.model = DiscreteDiffusionSDE(
            nn_diffusion=self.nn_diffusion,
            fix_mask=None,
            diffusion_steps=model_config.diffusion.steps
        )

    def forward(self, trajectories, conditions):
        B, T, D = trajectories.shape
        # Flatten [B, T, D] to [B, T*D] for MLP backbone
        flat_trajectories = trajectories.view(B, -1)
        return self.model.loss(flat_trajectories, conditions)

    def sample(self, conditions, n_samples=1):
        # Samples are flat [B, T*D]
        flat_samples = self.model.sample(conditions, n_samples=n_samples)
        # Unflatten to [B, T, D]
        return flat_samples.view(flat_samples.shape[0], self.horizon, self.obs_dim + self.action_dim)
