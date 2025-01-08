import heapq
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchinfo
from tqdm.auto import tqdm
import wandb

class Trainer:
    def __init__(self, configs):
        self.configs = configs
        # Setup device
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                self.device = torch.device("cuda")
                self.device_type = "multi"
            else:
                self.device = torch.device("cuda:0")
                self.device_type = "single"
        else:
            self.device = torch.device("cpu")
            self.device_type = "cpu"

    def set_train(self):
        """Convert all models to training mode.
        """
        self.model.train()

    def set_eval(self):
        """Convert all models to evaluation mode.
        """
        self.model.eval()

    def fit(self, model, datamodule):
        """Fitting the model with the data.
        """
        # Model
        self.model = model
        self.model.to(device=self.device)

        # Data loader
        datamodule.setup(stage="fit")
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()

        # Model summary
        torchinfo.summary(
            model=self.model,
            input_size=(
                8, 3,
                self.configs["transforms"]["input_heigh"],
                self.configs["transforms"]["input_width"]
            ),
            col_names = ("input_size", "output_size")
        )

        # Start fitting stage
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float('inf')
        num_of_epochs = self.configs["training"]["epoch"]
        print("======== BEGIN FITTING STAGE ========")
        for self.epoch in range(num_of_epochs):
            tqdm.write(f"==> Epoch [{self.epoch}/{num_of_epochs}]")
            self.run_epoch()

    def val(self):
        """Validating and save model if the performance is improved
        """
        val_loss = []
        best_heap = []
        worst_heap = []
        with torch.no_grad():
            with tqdm(self.val_dataloader, unit="batch", desc="Validating") as tloader:
                for sample_batch in tloader:
                    sample_batch["left_image"]=sample_batch["left_image"].to(self.device)
                    sample_batch["right_image"]=sample_batch["right_image"].to(self.device)
                    output_disps, losses = self.model.validation_step(sample_batch)

                    # Keep track batch losses
                    batch_loss = losses.cpu().numpy()
                    val_loss.extend(batch_loss)

                    # Keep track of 3 best cases and 3 worst cases
                    for i, loss in enumerate(batch_loss):
                        # Track best
                        heapq.heappush(best_heap, (-loss, sample_batch["raw_left_image"][i].cpu(), output_disps[i].cpu()))
                        if len(best_heap) > 3:
                            heapq.heappop(best_heap)
                        # Track worst
                        heapq.heappush(worst_heap, (loss, sample_batch["raw_left_image"][i].cpu(), output_disps[i].cpu()))
                        if len(worst_heap) > 3:
                            heapq.heappop(worst_heap)

            # Sort best and worst case base on losses
            best_heap = sorted([(-x[0], x[1], x[2]) for x in best_heap], key=lambda x: x[0])
            worst_heap = sorted(worst_heap, key=lambda x: x[0])
            
            # Calculate average validation loss
            avg_val_loss = np.mean(val_loss)

            # Save best-performing model
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.model.store_state(is_best=True)

            # Visualization
            fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(30,9), constrained_layout=True)
            for i in range(3):
                # Best case
                axs[0, i].imshow(best_heap[i][1].permute((1, 2, 0)).numpy())
                axs[0, i].set_title(f"Bestcase {i+1} - Loss: {best_heap[i][0]:.4f}")
                axs[0, i].axis("off")

                axs[1, i].imshow(best_heap[i][2].squeeze().numpy(), cmap="plasma")
                axs[1, i].axis("off")
                
                # Worst case
                axs[2, i].imshow(worst_heap[i][1].permute((1, 2, 0)).numpy())
                axs[2, i].set_title(f"Worstcase {i+1} - Loss: {worst_heap[i][0]:.4f}")
                axs[2, i].axis("off")

                axs[3, i].imshow(worst_heap[i][2].squeeze().numpy(), cmap="plasma")
                axs[3, i].axis("off")

            ## Log to WandB
            wandb.log({
                "Validation loss": avg_val_loss,
                "Best validation loss": self.best_val_loss,
                "Best and worst cases": wandb.Image(fig),
            }, step=self.step)

    def run_epoch(self):
        """Run a single epoch constitute include training and validation
        """
        with tqdm(self.train_dataloader, unit="batch", desc="Training") as tloader:
            for sample_batch in tloader:
                sample_batch["left_image"]=sample_batch["left_image"].to(self.device)
                sample_batch["right_image"]=sample_batch["right_image"].to(self.device)

                # Pass input batch through model to get output images and loss value.
                loss = self.model.training_step(sample_batch)

                # Back Propagation
                self.model.optim.zero_grad()
                loss.backward()
                self.model.optim.step()

                # Logging
                if (self.step+1) % self.configs["logging"]["log_frequency"] == 0:
                    wandb.log({
                        "Training loss": loss.item(),
                    }, step=self.step)
                
                self.step += 1

        self.set_eval()
        self.val()
        self.set_train()
