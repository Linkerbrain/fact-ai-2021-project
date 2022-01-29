from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class AffineTransform(nn.Module):
    def __init__(self, clip_transX, clip_transY, clip_shearX, clip_shearY, clip_scaleX, clip_scaleY):
        super(AffineTransform, self).__init__()
        clip_matrix = torch.FloatTensor([[clip_scaleX, clip_shearX, clip_transX],
                                         [clip_shearY, clip_scaleX, clip_transY]])
        identity = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float)
        self.clip_min = identity - clip_matrix
        self.clip_max = identity + clip_matrix
        # Initialize with identity transformation
        self.theta = nn.Parameter(self.identity.clone(), requires_grad=True)

    # Spatial transformer network forward function
    def stn(self, x):
        grid = F.affine_grid(self.theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x

    def clip(self):
        self.theta.data.clamp_(self.clip_min, self.clip_max)

def translate(x, shift):
    theta = torch.zeros(x.size(0), 2, 3).to(x.device)
    theta[0, 0, 0] = 1
    theta[0, 1, 1] = 1
    theta[0, :, 2] = shift
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False)
    return x


class Translation(nn.Module):
    def __init__(self, clipX=2.0, clipY=2.0):
        """
        @param clipX, clipY: max translation range, 2.0 allows for any distiguishable translation
        (a distance of 2.0 shifts the image out of the frame completely)
        """
        super(Translation, self).__init__()

        # Initialize with identity translation
        self.xy = nn.Parameter(torch.tensor([[0, 0]], dtype=torch.float))
        self.clipX = clipX
        self.clipY = clipY
        self._theta = None

    # Spatial transformer network forward function
    def stn(self, x):
        return translate(x, self.xy)

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x

    def clip(self):
        """Called after each training step to limit the translation range"""
        self.xy.data[0, 0].clamp_(min=-self.clipX, max=self.clipX)
        self.xy.data[0, 1].clamp_(min=-self.clipY, max=self.clipY)
