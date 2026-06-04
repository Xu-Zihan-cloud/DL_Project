import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb

from src.utils.cpu_optimizer import optimize_cpu_performance, get_autocast_context
from src.datamodules.d4rl_dataset import get_dataloader
from src.models.diffuser_module import DecisionDiffuserWrapper

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    # 1. Hardware Optimization
    optimize_cpu_performance(threads_per_process=32)
    device = torch.device(cfg.device)
    autocast_ctx = get_autocast_context()

    # 2. WandB Logging
    if cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.env.name}-baseline"
        )

    # 3. Data Loading
    dataloader = get_dataloader(
        env_name=cfg.env.name,
        horizon=cfg.env.horizon,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.num_workers
    )

    # 4. Model & Optimizer
    model = DecisionDiffuserWrapper(
        obs_dim=cfg.env.obs_dim,
        action_dim=cfg.env.action_dim,
        horizon=cfg.env.horizon,
        model_config=cfg.model
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.model.lr, 
        weight_decay=cfg.model.weight_decay
    )

    # 5. Training Loop
    model.train()
    for epoch in range(cfg.epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            trajectories = batch["trajectories"].to(device)
            conditions = batch["returns"].to(device)
            
            optimizer.zero_grad()
            
            with autocast_ctx:
                loss = model(trajectories, conditions)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        if cfg.wandb.mode != "disabled":
            wandb.log({"train/loss": avg_loss, "epoch": epoch})
        
        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train()
