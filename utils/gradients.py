import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sobel(nn.Module):
    """
    Computes the spatial gradients of 3D data using Sobel filters.
    """

    def __init__(self, device):
        super().__init__()
        self.pad = nn.ReplicationPad2d(1)
        a = np.zeros((1, 1, 3, 3))
        b = np.zeros((1, 1, 3, 3))
        a[0, :, :, :] = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        b[0, :, :, :] = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        self.a = torch.from_numpy(a).float().to(device)
        self.b = torch.from_numpy(b).float().to(device)

    def forward(self, x):
        """
        :param x: [batch_size x 1 x H x W] input tensor
        :return gradx: [batch_size x 2 x H x W-1] spatial gradient in the x direction
        :return grady: [batch_size x 2 x H-1 x W] spatial gradient in the y direction
        """

        x = x.view(-1, 1, x.shape[2], x.shape[3])  # (batch * channels, 1, height, width)
        x = self.pad(x)
        gradx = F.conv2d(x, self.a, groups=1) / 8  # normalized gradients
        grady = F.conv2d(x, self.b, groups=1) / 8  # normalized gradients
        return gradx, grady
