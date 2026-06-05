import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb, os, gym, d4rl, numpy as np, logging, sys, pandas as pd

# Professional logging setup for remote headless servers
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.utils.cpu_optimizer import optimize_cpu_performance, get_autocast_context
from src.datamodules.d4rl_dataset import D4RLTrajectoryDataset
from src.models.diffuser_module import DecisionDiffuserWrapper

def evaluate(model, dataset, env_name, target_rtg, n_episodes=10):
    """
    Performs Mujoco Rollouts to evaluate the trained model with proper normalization and clipping.
    """
    model.eval()
    env = gym.make(env_name)
    device = next(model.parameters()).device
    total_rewards = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 1. Normalize Observation
            norm_obs = torch.from_numpy(dataset.normalize_obs(obs)).float().to(device)
            
            # 2. Prepare Condition (RTG)
            conditions = torch.tensor([[target_rtg]]).float().to(device)
            
            # 3. Sample Trajectory
            with torch.no_grad():
                samples = model.sample(conditions) # [1, T, Obs+Act]
                
            # 4. Extract first action, unnormalize, and clip
            sampled_action = samples[0, 0, model.obs_dim:].cpu().numpy()
            action = dataset.unnormalize_action(sampled_action)
            action = np.clip(action, -1, 1) # Critical for Mujoco stability
            
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            
        total_rewards.append(episode_reward)
    
    avg_reward = np.mean(total_rewards)
    norm_score = d4rl.get_normalized_score(env_name, avg_reward) * 100
    return avg_reward, norm_score

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    # 1. Hardware Optimization (CPU 512-thread)
    optimize_cpu_performance(threads_per_process=32)
    device = torch.device(cfg.device)
    autocast_ctx = get_autocast_context()

    # 2. Directory Initialization
    work_dir = hydra.utils.get_original_cwd()
    result_dir = os.path.join(work_dir, "results", cfg.env.name)
    os.makedirs(result_dir, exist_ok=True)

    # 3. Initialization
    if cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project, 
            name=f"{cfg.env.name}-academic-baseline", 
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    # 4. Data Loading (Normalized & Scaled)
    dataset = D4RLTrajectoryDataset(env_name=cfg.env.name, horizon=cfg.env.horizon)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.train_batch_size, shuffle=True, 
        num_workers=cfg.num_workers, pin_memory=True, prefetch_factor=4
    )

    # 5. Model & Optimizer
    model = DecisionDiffuserWrapper(
        obs_dim=cfg.env.obs_dim, action_dim=cfg.env.action_dim,
        horizon=cfg.env.horizon, model_config=cfg.model
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.model.lr, weight_decay=cfg.model.weight_decay)

    # 6. Training & Logging
    history = []
    best_score = -float('inf')
    logger.info(f"Training started. Results will be saved in: {result_dir}")

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", file=sys.stdout)
        
        for batch in pbar:
            trajectories = batch["trajectories"].to(device)
            conditions = batch["returns"].to(device)
            
            optimizer.zero_grad()
            with autocast_ctx:
                loss = model(trajectories, conditions)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_loss = epoch_loss / len(dataloader)
        
        # 7. Periodic Evaluation & Persistence
        if epoch % cfg.eval_freq == 0 or epoch == cfg.epochs - 1:
            avg_reward, norm_score = evaluate(model, dataset, cfg.env.name, cfg.env.target_rtg)
            logger.info(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Reward: {avg_reward:.2f} | Score: {norm_score:.2f}")
            
            # Save history to CSV for standardized reporting
            res = {"epoch": epoch, "loss": avg_loss, "reward": avg_reward, "score": norm_score}
            history.append(res)
            pd.DataFrame(history).to_csv(os.path.join(result_dir, "progress.csv"), index=False)
            
            if cfg.wandb.mode != "disabled":
                wandb.log(res)

            # Track Best Model
            if norm_score > best_score:
                best_score = norm_score
                save_path = os.path.join(result_dir, "best_model.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'stats': dataset.stats,
                    'epoch': epoch,
                    'score': norm_score
                }, save_path)
                logger.info(f"New Best Score: {best_score:.2f}, model saved to {save_path}")

        # Regular Latest Save
        if epoch % cfg.save_freq == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'stats': dataset.stats,
                'epoch': epoch
            }, os.path.join(result_dir, "latest.pt"))

if __name__ == "__main__":
    train()
