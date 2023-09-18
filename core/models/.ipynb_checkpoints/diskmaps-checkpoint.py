import torch
from torch import nn as nn
import numpy as np


class DistMaps(nn.Module):
    def __init__(self, norm_radius, spatial_scale=1.0, use_disks=False):
        super(DistMaps, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.use_disks = use_disks


    def get_coord_features(self, points, batchsize, rows, cols):

        # points' shape is (transform_num, 2*max(postive_clicks_num, negative_clicks_num), 3)
        #num_points = points.shape[1] # not the number of clicks, but max(positive_click_num, negative click_num)
        # B maxnum 3
        points = points.view(-1, points.size(2))
        # turn points' shape to (points.size(0), 3), points.size(0)=transform_num*2*num_points
        points, points_class = torch.split(points, [2, 1], dim=1)
        # points' shape is (points.size(0), 2), points.size(0)=transform_num*2*num_points

        invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0 # find (-1,-1,-1) which is not click
        # invalid_points' shape is [points.size(0)]
        row_array = torch.arange(start=0, end=rows, step=1, dtype=torch.float32, device=points.device)
        col_array = torch.arange(start=0, end=cols, step=1, dtype=torch.float32, device=points.device)

        coord_rows, coord_cols = torch.meshgrid(row_array, col_array)
        coords = torch.stack((coord_rows, coord_cols), dim=0).unsqueeze(0).repeat(points.size(0), 1, 1, 1)
        # coods' shape is (points.size(0), 2, rows, cols)
        add_xy = (points * self.spatial_scale).view(points.size(0), points.size(1), 1, 1)
        # turn points*scale's shape to (points.size(0), 2, 1, 1)
        coords.add_(-add_xy)
        if not self.use_disks:
            coords.div_(self.norm_radius * self.spatial_scale)
        coords.mul_(coords)

        coords[:, 0] += coords[:, 1] #(x-point)**2+(y-point)**2
        coords = coords[:, :1]

        coords[invalid_points, :, :, :] = 1e6 # make (-1,-1,-1) largest (only support radius<=100)

        coords = coords.view(-1, num_points, 1, rows, cols) # ->(2*tranform),num_points,1,h,w
        coords = coords.min(dim=1)[0]  # -> (bs * num_masks * 2) x 1 x h x w
        # sign every point on the grid with the click point nearest.
        coords = coords.view(-1, 2, rows, cols)
        if self.use_disks:
            coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2).float()
        else:
            coords.sqrt_().mul_(2).tanh_() # tanh(2*sqrt( [(x-x0)^2+(y-y0)^2]/[radius*spatial_sacle] ))

        return coords

    def forward(self, x, coords, pred):
        return self.get_coord_features(coords, pred, x.shape[0], x.shape[2], x.shape[3])

