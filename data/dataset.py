import os

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.v2 as transforms

class KITTIDataset(Dataset):
    def __init__(self, data_dir=str, train=True, transforms=None):
        super().__init__()
        self.data_dir = data_dir
        self.train = train
        self.transforms = transforms
        self.samples = []

        # Loop through all images to get pair of image paths.
        for date in os.listdir(self.data_dir):
            date_dir = os.path.join(self.data_dir, date)
            if not os.path.exists(date_dir) or os.path.isfile(data_dir):
                continue
            for scene in os.listdir(date_dir):
                scene_dir = os.path.join(date_dir, scene, "image_02", "data")
                if not os.path.exists(scene_dir) or os.path.isfile(scene_dir):
                    continue
                for left_image in os.listdir(scene_dir):
                    left_image_path = os.path.join(scene_dir, left_image)
                    right_image_path = left_image_path.replace("image_02", "image_03")
                    if not os.path.exists(right_image_path) or not left_image_path.endswith(".png"):
                        continue
                    self.samples.append((left_image_path, right_image_path))

        # If train dataset is require, return first 80% of data.
        # Train dataset composed of training and validation subset.
        # Else, return evaluation set composed of the last 20% of data. 
        if train:
            self.samples=self.samples[:int(len(self.samples)*0.8)]
        else:
            self.samples=self.samples[int(len(self.samples)*0.8+1):]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        left_image = Image.open(self.samples[index][0])
        right_image = Image.open(self.samples[index][1])

        # Transform images
        if self.transforms is not None:
            transformed_left_image = self.transforms(left_image)
            transformed_right_image = self.transforms(right_image)

        left_image = transforms.Compose([
            transforms.Resize((transformed_left_image.shape[1:])),
            transforms.ToImage(),
            transforms.ToDtype(dtype=torch.float32, scale=True)
        ])(left_image)

        return {"raw_left_image":left_image, "left_image":transformed_left_image, "right_image":transformed_right_image}