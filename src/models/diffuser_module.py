import torch
import torch.nn as nn
from CleanDiffuser.models.diffusion import DiffusionModel
from CleanDiffuser.nn_diffusion import MLP, DiT

class DecisionDiffuserWrapper(nn.Module):
    def __init__(self, obs_dim, action_dim, horizon, model_config):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        
        # In CleanDiffuser, Decision Diffuser often uses a U-Net or MLP
        # Here we follow the core DD implementation logic
        input_dim = obs_dim + action_dim
        
        # CleanDiffuser setup (Simplified representation for baseline)
        self.model = DiffusionModel(
            nn_diffusion=MLP(
                d_in=input_dim, 
                d_out=input_dim, 
                d_model=model_config.nn_diffusion.d_model,
                n_layers=model_config.nn_diffusion.n_layers
            ),
            fix_mask=None, # DD usually doesn't fix mask for full trajectory
            loss_type="l2",
            diffusion_steps=model_config.diffusion.steps,
            beta_schedule=model_config.diffusion.beta_schedule
        )

    def forward(self, trajectories, conditions):
        return self.model.loss(trajectories, conditions)

    def sample(self, conditions, n_samples=1):
        # Initial sampling using DDPM baseline
        return self.model.sample(conditions, n_samples=n_samples)
