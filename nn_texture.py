import torch, torch.nn as nn, torch.nn.functional as F
from utils import *

class FeatureGrid(nn.Module):
    def __init__(self, resolutions=(16, 32, 64, 128), feat_dim=2):
        super().__init__()
        # TODO: one learnable grid per resolution, each shaped (1, feat_dim, R, R)

        # nn.ParameterList registers the params (learnable vals) with the model -> model.parameters() 
        self.grids = nn.ParameterList([
            nn.Parameter(torch.randn(1, feat_dim, R, R) * 0.01) # small initialization
            for R in resolutions
        ])
        self.feat_dim = feat_dim
        self.out_dim = feat_dim * len(resolutions) # concat interpolated feat vectors for all resolutions
        self.resolutions = resolutions

    def forward(self, uv):  
        '''
        N = batch size

        Input: uv --> (N, 2) in [0, 1]

        TODO: bilinear-sample each grid at uv and concatenate the features across resolutions -> (N, out_dim)
        '''

        # grid_sample takes [-1,1] coords
        grid_coords = uv * 2.0 - 1.0
        # treat coord tensor as Nx1 output img: (batch, out_H, out_W, 2) -> (1, N, 1, 2)
        grid_coords = grid_coords.view(1, -1, 1, 2) 

        features = []
        for grid in self.grids:
            sampled = F.grid_sample(
                grid,
                grid_coords,
                mode="bilinear",
                padding_mode="border",
                align_corners=False
            )

            # sampled shape: (1, feat_dim, N, 1) --> turn to (N, feat_dim)
            sampled = sampled.squeeze(0)   # (feat_dim, N, 1)
            sampled = sampled.squeeze(-1)  # (feat_dim, N)
            sampled = sampled.T            # (N, feat_dim)

            features.append(sampled)

        # concat across resolutions (N, feat_dim * n_resolutions)
        return torch.cat(features, dim=1)

class ColorMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # TODO: 2 hidden layers of width 64 (Linear + ReLU), then Linear -> 3 and a Sigmoid (RGB in [0, 1])
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), 
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
            return self.net(x)


# full neural texture compression model
class NeuralTexture(nn.Module):
    def __init__(self, resolutions, feat_dim):
        super().__init__()
        self.grid = FeatureGrid(resolutions=resolutions, feat_dim=feat_dim)
        self.mlp  = ColorMLP(self.grid.out_dim)

    def forward(self, uv):
        # 1. send uv through featuregrid to get feature vector
        # 2. send feature vector through mlp 
        return self.mlp(self.grid(uv))
    


    