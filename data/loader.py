from .dataset import KITTIDataset

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms.v2 as transforms

class DataLoaderModule():
    def __init__(self, configs):
        self.data_dir = configs["dataset"]["path"]
        self.batch_size = configs["training"]["batch_size"]
        self.input_size = (configs["transforms"]["input_heigh"], configs["transforms"]["input_width"])
        self.augmentation = [
            transforms.ColorJitter(
                configs["transforms"]["color_jitter"]["brightness"],
                configs["transforms"]["color_jitter"]["contrast"],
                configs["transforms"]["color_jitter"]["saturation"],
                configs["transforms"]["color_jitter"]["hue"]
            )
        ]

    def setup(self, stage):
        """Prepair dataset object for each stage
        """
        # Define data transform with/without augmentation
        data_transform = [transforms.Resize(self.input_size)]
        data_transform += self.augmentation if stage=="fit" else [transforms.Lambda(lambda x: x)]
        data_transform += [
            transforms.ToImage(),
            transforms.ToDtype(dtype=torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        data_transform = transforms.Compose(data_transform)

        if stage == "fit":
            training_dataset = KITTIDataset(data_dir=self.data_dir, train=True, transforms=data_transform)
            self.train_set, self.valid_set = random_split(dataset=training_dataset, lengths=[0.75, 0.25])
        elif stage == "test":
            self.test_set = KITTIDataset(data_dir=self.data_dir, train=False, transforms=data_transform)
        elif stage == "predict":
            self.pred_set = KITTIDataset(data_dir=self.data_dir, train=False, transforms=data_transform)
    
    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_set, batch_size=self.batch_size, drop_last=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(dataset=self.pred_set, batch_size=self.batch_size)