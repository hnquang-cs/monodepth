# utils/logger.py

import wandb

def init_wandb(configs):
    wandb.init(
        project=configs["project"]["name"],
        name=configs["project"]["experiment"],
        config={
            "dataset": configs["dataset"]["name"],
            "batch_size": configs["training"]["batch_size"],
            "optimizer": configs["training"]["optimizer"],
            "learning-rate": configs["training"]["lr"]
        }
    )