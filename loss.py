import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, kernel_size=31, smooth=.001, activation=nn.Sigmoid()):
        super(DiceLoss, self).__init__()
        self.activation = activation
        self.smooth = smooth
        self.kernel_size=kernel_size

    def forward(self, output, target):
        output = self.activation(output)
        weight = 1 + 5 * torch.abs(
            F.avg_pool2d(target, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2) - target)
        inter = ((output * target) * weight).sum(dim=(2, 3))
    
        # Reshape directly to (batch_size, 1) due to having only the single channel
        inter = inter.view(-1, 1)
        mask_sum = (target + output) * weight
        mask_sum = mask_sum.sum(dim=(2, 3)).view(-1, 1)
        
        loss = 1 - (2 * inter + self.smooth) / (mask_sum + self.smooth)
        return loss.mean()
    
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, activation=nn.Sigmoid()):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        
        # Set reduction to False to obtain the probability of each cls
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        return self.loss(output, target)


class WeightedIOULoss(nn.Module):
    def __init__(self, kernel_size=31, activation=nn.Sigmoid()):
        super(WeightedIOULoss, self).__init__()
        self.activation = activation
        self.kernel_size = kernel_size
        
    def forward(self, output, target):
        output = self.activation(output)
        weight = 1 + 5 * torch.abs(
            F.avg_pool2d(target, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2) - target)
        inter = ((output * target) * weight).sum(dim=(2, 3))
        union = ((output + target) * weight).sum(dim=(2, 3))
        loss = 1 - (inter + 1) / (union - inter + 1)
        
        return loss.mean()
