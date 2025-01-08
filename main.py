import argparse

from data.loader import DataLoaderModule
from model import DepthEstimationModel
from trainer import Trainer
from utils.logger import init_wandb

import yaml
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/default.yaml", help="Path to the YAML configuration file")
    parser.add_argument("--mode", type=int, choices=[0, 1, 2], help="Mode flag. Training, Evaluation, Prediction mode corresponding to 0, 1, 2 respectively")
    parser.add_argument("--resume", type=bool, default=False, help="Resume training flag")
    args = parser.parse_args()

    with open(args.config) as f:
        configs = yaml.safe_load(f)

    init_wandb(configs)

    if args.mode == 0:
        model = DepthEstimationModel(configs=configs)
        if args.resume:
            model.load_state(is_best=False)
        dataloader = DataLoaderModule(configs=configs)
        trainer = Trainer(configs)
        trainer.fit(model=model, datamodule=dataloader)

    if args.mode == 1:
        #TODO
        # Evaluator(configs)
        pass

    wandb.finish()