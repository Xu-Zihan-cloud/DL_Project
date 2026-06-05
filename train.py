import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb, os, gym, d4rl, numpy as np, logging, sys, pandas as pd

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

from src.utils.cpu_optimizer import optimize_cpu_performance, get_autocast_context
from src.datamodules.d4rl_dataset import D4RLTrajectoryDataset
from src.models.diffuser_module import DecisionDiffuserWrapper

def evaluate(model, dataset, env_name, target_rtg, n_episodes=10):
    model.eval()
    env = gym.make(env_name)
    device = next(model.parameters()).device
    total_rewards = []
    
    for _ in range(n_episodes):
        obs, done, ep_rew = env.reset(), False, 0
        while not done:
            norm_obs = torch.from_numpy(dataset.normalize_obs(obs)).float().to(device)
            # Conditions is [1, 1]
            conditions = torch.tensor([[target_rtg]]).float().to(device)
            with torch.no_grad():
                samples = model.sample(conditions) 
            sampled_action = samples[0, 0, model.obs_dim:].cpu().numpy()
            action = np.clip(dataset.unnormalize_action(sampled_action), -1, 1)
            obs, reward, done, _ = env.step(action)
            ep_rew += reward
        total_rewards.append(ep_rew)
    
    avg_reward = np.mean(total_rewards)
    norm_score = d4rl.get_normalized_score(env_name, avg_reward) * 100
    return avg_reward, norm_score

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    optimize_cpu_performance(threads_per_process=32)
    device = torch.device(cfg.device)
    autocast_ctx = get_autocast_context()

    work_dir = hydra.utils.get_original_cwd()
    result_dir = os.path.join(work_dir, "results", cfg.env.name)
    os.makedirs(result_dir, exist_ok=True)

    if cfg.wandb.mode != "disabled":
        wandb.init(project=cfg.wandb.project, name=f"{cfg.env.name}-v2-final", config=OmegaConf.to_container(cfg, resolve=True))

    dataset = D4RLTrajectoryDataset(env_name=cfg.env.name, horizon=cfg.env.horizon)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.train_batch_size, shuffle=True, 
        num_workers=cfg.num_workers, pin_memory=True, prefetch_factor=4
    )

    model = DecisionDiffuserWrapper(
        obs_dim=cfg.env.obs_dim, action_dim=cfg.env.action_dim,
        horizon=cfg.env.horizon, model_config=cfg.model
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.model.lr, weight_decay=cfg.model.weight_decay)

    history, best_score = [], -float('inf')
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", file=sys.stdout)
        
        for batch in pbar:
            trajectories, conditions = batch["trajectories"].to(device), batch["returns"].to(device)
            optimizer.zero_grad()
            with autocast_ctx:
                loss = model(trajectories, conditions)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        if epoch % cfg.eval_freq == 0 or epoch == cfg.epochs - 1:
            target_rtg = cfg.env.get("target_rtg", 1.0)
            avg_rew, score = evaluate(model, dataset, cfg.env.name, target_rtg)
            logger.info(f"Epoch {epoch} | Loss: {epoch_loss/len(dataloader):.6f} | Score: {score:.2f}")
            
            res = {"epoch": epoch, "loss": epoch_loss/len(dataloader), "reward": avg_rew, "score": score}
            history.append(res)
            pd.DataFrame(history).to_csv(os.path.join(result_dir, "progress.csv"), index=False)
            
            if score > best_score:
                best_score = score
                torch.save({'model': model.state_dict(), 'stats': dataset.stats}, os.path.join(result_dir, "best_model.pt"))

        if epoch % cfg.save_freq == 0:
            torch.save({'model': model.state_dict(), 'stats': dataset.stats}, os.path.join(result_dir, "latest.pt"))

if __name__ == "__main__":
    train()
