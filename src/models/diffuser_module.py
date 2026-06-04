import torch
import torch.nn as nn
from cleandiffuser.diffusion import DiffusionModel
from cleandiffuser.nn_diffusion.mlps import MlpNNDiffusion
from cleandiffuser.nn_diffusion.dit import DiT1d

class DecisionDiffuserWrapper(nn.Module):
    def __init__(self, obs_dim, action_dim, horizon, model_config):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        
        # In cleandiffuser, Decision Diffuser often uses a U-Net or MLP
        # Here we follow the core DD implementation logic
        input_dim = obs_dim + action_dim
        
        # cleandiffuser setup (Simplified representation for baseline)
        self.model = DiffusionModel(
            nn_diffusion=MlpNNDiffusion(
                x_dim=input_dim, 
                hidden_dims=[model_config.nn_diffusion.d_model] * model_config.nn_diffusion.n_layers
            ),
            fix_mask=None, # DD usually doesn't fix mask for full trajectory
            diffusion_steps=model_config.diffusion.steps
        )

    def forward(self, trajectories, conditions):
        return self.model.loss(trajectories, conditions)

    def sample(self, conditions, n_samples=1):
        # Initial sampling using DDPM baseline
        return self.model.sample(conditions, n_samples=n_samples)
