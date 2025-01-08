import os

from .loss_module import ReprojectionLoss, DisparitySmoothnessLoss
from .networks.encoder import ResNet
from .networks.decoder import Decoder

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthEstimationModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        # Initialize networks
        self.encoder = ResNet(in_channels=3)
        self.decoder = Decoder()

        # Loss modules
        self.ap_loss = ReprojectionLoss()
        self.ds_loss = DisparitySmoothnessLoss()

        # Optimizer
        params_to_train = []
        params_to_train += list(self.encoder.parameters())
        params_to_train += list(self.decoder.parameters())
        if configs["training"]["optimizer"] == "Adam":
            self.optim = torch.optim.Adam(params=params_to_train, lr=configs["training"]["lr"])
        elif configs["training"]["optimizer"] == "SGD":
            self.optim = torch.optim.SGD(params=params_to_train, lr=configs["training"]["lr"])
        else:
            self.optim = torch.optim.Adam(params=params_to_train, lr=configs["training"]["lr"])

    def forward(self, x):
        features = self.encoder(x)
        disparities = self.decoder(features)
        return disparities

    def load_state(self, is_best=True):
        load_dir = os.path.join(self.configs["logging"]["checkpoint_path"], "best" if is_best else "last")
        if not os.path.exists(load_dir):
            os.makedirs(load_dir)
        # Load params
        self.encoder.load_state_dict(torch.load(f"{load_dir}/encoder.pt"))
        self.decoder.load_state_dict(torch.load(f"{load_dir}/decoder.pt"))
        self.optim.load_state_dict(torch.load(f"{load_dir}/optim.pt"))
    
    def store_state(self, is_best=True):
        save_dir = os.path.join(self.configs["logging"]["checkpoint_path"], "best" if is_best else "last")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Save params
        torch.save(self.encoder.state_dict(), f"{save_dir}/encoder.pt")
        torch.save(self.decoder.state_dict(), f"{save_dir}/decoder.pt")
        torch.save(self.optim.state_dict(), f"{save_dir}/optim.pt")

    def training_step(self, batch):
        # Passing input batch though networks
        disparities = self.forward(batch["left_image"])

        # Upscale disparity maps
        loss = 0
        for i in range(len(disparities)):
            disparities[i] = torch.nn.functional.interpolate(
                input=disparities[i],
                scale_factor=2**(2-i),
                mode="bilinear"
            )
            # Generate reprojected left-view from right-view images
            wrapped_left_imgs = self.wrap_image(disparities=disparities[i], images=batch["right_image"])

            loss += self.ap_loss(wrapped_left_imgs, batch["left_image"])
            loss += (0.1/(2**(2-1)))*self.ds_loss(disparities[i], batch["left_image"])
        return loss
    
    def validation_step(self, batch):
        # Passing input batch though networks
        disparities = self.forward(batch["left_image"])

        # Upscale disparity maps
        loss = 0
        for i in range(len(disparities)):
            disparities[i] = torch.nn.functional.interpolate(
                input=disparities[i],
                scale_factor=2**(2-i),
                mode="bilinear"
            )
            # Generate reprojected left-view from right-view images
            wrapped_left_imgs = self.wrap_image(disparities=disparities[i], images=batch["right_image"])

            loss += self.ap_loss(wrapped_left_imgs, batch["left_image"], separate=True)
            loss += (0.1/(2**(2-1)))*self.ds_loss(disparities[i], batch["left_image"], separate=True)
        return disparities[2], loss
        
    def wrap_image(self, disparities, images):
        img_height, img_width = images.shape[2:4]
        h_flows = -disparities
        v_flows = torch.nn.Parameter(torch.zeros((1, img_height, img_width))).to(device=disparities.device)
        v_flows = v_flows * h_flows

        # coordinate grid, normalized to [-1, 1] to fit into grid_sample
        coord_x = np.tile(range(img_width), (img_height, 1)) / ((img_width-1)/2) - 1
        coord_y = np.tile(range(img_height), (img_width, 1)).T / ((img_height-1)/2) - 1
        grids = np.stack([coord_x, coord_y])
        grids = torch.Tensor(grids).permute(1,2,0)
        grids = torch.nn.Parameter(grids, requires_grad=False).to(device=disparities.device)

        # transformation
        trans = torch.cat([h_flows, v_flows], dim=1)
        wrap_grids = grids + trans.permute(0,2,3,1)

        # warping
        wrap_imgs = F.grid_sample(images, wrap_grids, padding_mode="border", align_corners=False)

        return wrap_imgs

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())