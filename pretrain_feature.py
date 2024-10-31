import hydra
import jax.numpy as jnp
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.rich import tqdm

from environments import make_env_and_dataset


@hydra.main(version_base=None, config_path="configs/", config_name="dispo.yaml")
def train(config):
    # Initialize wandb
    wandb.init(
        project="all-task-rl",
        group=config.env_id,
        job_type=f"{config.algo}_train_feature",
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    )

    # Make environment and dataset
    env, dataset = make_env_and_dataset(
        config.env_id, config.seed, config.feat.type, config.feat.dim
    )

    # Build dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        drop_last=True,
    )

    # Round steps to epochs
    num_epochs = config.training.num_steps // len(dataloader)
    num_steps = num_epochs * len(dataloader)
    OmegaConf.update(config, "training.num_steps", num_steps)

    # Learn features
    step = 0
    pbar = tqdm(total=config.training.num_steps)
    for epoch in range(num_epochs):
        for batch in dataloader:
            batch = {k: jnp.array(v) for k, v in batch.items()}

            # Update feature network
            train_info = env.train(
                obs=batch["observations"],
                act=batch["actions"],
                next_obs=batch["next_observations"],
            )
            wandb.log(train_info)

            # Save checkpoint
            if (step + 1) % config.training.save_every == 0:
                env.save()

            step += 1
            pbar.update(1)

        # Logging
        wandb.log({"train/epoch": epoch})
    pbar.close()


if __name__ == "__main__":
    train()
