import torch
from torch import nn


class tEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, dim // 2), nn.SiLU(), nn.Linear(dim // 2, dim)
        )

    def forward(self, t):
        return self.net(t)


class xEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, dim // 2), nn.SiLU(), nn.Linear(dim // 2, dim)
        )

    def forward(self, x):
        return self.net(x)


class VelocityNet(nn.Module):
    def __init__(self, z_dim: int, x_embed_dim: int, t_embed_dim: int, channels: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(z_dim + x_embed_dim + t_embed_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, z_dim),
        )

    def forward(self, z, x_embedding, t_embedding):
        return self.net(torch.cat((z, x_embedding, t_embedding), dim=-1))
