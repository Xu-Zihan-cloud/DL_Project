import torch
import torch.nn as nn
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_diffusion.mlps import MlpNNDiffusion

class DecisionDiffuserWrapper(nn.Module):
    def __init__(self, obs_dim, action_dim, horizon, model_config):
        super().__init__()
        self.obs_dim, self.action_dim, self.horizon = obs_dim, action_dim, horizon
        
        # Augmented State Approach:
        # We append the RTG condition (dim=1) to the end of the flattened trajectory.
        # This makes the model API-agnostic regarding conditioning.
        self.traj_dim = horizon * (obs_dim + action_dim)
        self.x_dim = self.traj_dim + 1 
        
        # 1. Setup unconditional MLP backbone
        self.nn_diffusion = MlpNNDiffusion(
            x_dim=self.x_dim, 
            hidden_dims=[512, 512, 512]
        )
        
        # 2. Create fix_mask to protect the RTG dimension from diffusion
        # 1 means fixed (known), 0 means diffused (unknown)
        fix_mask = torch.zeros(self.x_dim)
        fix_mask[-1] = 1.0
        self.register_buffer("fix_mask", fix_mask)
        
        # 3. Setup SDE with fix_mask
        self.model = DiscreteDiffusionSDE(
            nn_diffusion=self.nn_diffusion,
            fix_mask=self.fix_mask,
            diffusion_steps=model_config.diffusion.steps
        )

    def forward(self, trajectories, conditions):
        # trajectories: [B, T, D], conditions: [B, 1]
        B = trajectories.shape[0]
        flat_trajectories = trajectories.view(B, -1)
        # Augment: [B, T*D + 1]
        x0 = torch.cat([flat_trajectories, conditions], dim=-1)
        return self.model.loss(x0)

    def sample(self, conditions, n_samples=1):
        # conditions (target_rtg): [B, 1]
        B = conditions.shape[0]
        device = conditions.device
        
        # Create prior: [B, T*D + 1]
        # Trajectory part is noise, RTG part is the target condition
        noise = torch.randn((B, self.traj_dim), device=device)
        prior = torch.cat([noise, conditions], dim=-1)
        
        # Sample using the prior and fix_mask (masked dims stay fixed to prior values)
        # Note: cleandiffuser.sample returns (samples, log_probs)
        flat_samples, _ = self.model.sample(prior=prior)
        
        # Strip the RTG dimension and unflatten
        traj_samples = flat_samples[:, :self.traj_dim]
        return traj_samples.view(B, self.horizon, self.obs_dim + self.action_dim)
