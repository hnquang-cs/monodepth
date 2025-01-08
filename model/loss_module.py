import torch
import torch.nn as nn
import torch.nn.functional as F

class ReprojectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predicts, targets, separate=False):
        loss = F.mse_loss(input=predicts, target=targets, reduction="none")
        return torch.mean(loss, dim=[1,2,3] if separate else None)
    
class DisparitySmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, disparites, images, separate=False):
        # Calculate weights for individual pixels by 
        # applying reverse exponential function on image gradient.
        image_gradient_x = self.gradient_x(images)
        image_gradient_y = self.gradient_y(images)
        weights_x = torch.exp(-image_gradient_x)
        weights_y = torch.exp(-image_gradient_y)

        # Calculate loss by applying weights 
        # on disparity gradients map then get the mean value
        disparity_gradient_x = self.gradient_x(disparites)
        disparity_gradient_y = self.gradient_y(disparites)
        loss_x = disparity_gradient_x*weights_x
        loss_y = disparity_gradient_y*weights_y

        return torch.mean(loss_x + loss_y, dim=[1,2,3] if separate else None) # Batch is [B, C, H, W]

    def gradient_x(self, input):
        input_img = F.pad(input=input, pad=[0,1,0,0], mode="replicate")
        grad_x = input_img[:,:,:,1:] - input_img[:,:,:,:-1]
        return grad_x
    
    def gradient_y(self, input):
        input_img = F.pad(input=input, pad=[0,0,0,1], mode="replicate")
        grad_y = input_img[:,:,1:,:] - input_img[:,:,:-1,:]
        return grad_y