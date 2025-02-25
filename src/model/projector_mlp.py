import torch.nn as nn
import torch.nn.functional as F

class MLP_1024_to_256(nn.Module):
    def __init__(self, hidden_size, projector_dim, z_dim):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, projector_dim) 
        self.conv = nn.Conv1d(projector_dim, projector_dim, kernel_size=4, stride=4)
        self.linear2 = nn.Linear(projector_dim, z_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.silu(x)
        x = x.permute(0, 2, 1) # (N, T, D) -> (N, D, T)
        x = self.conv(x) # (N, D, T) -> (N, D, T/4)
        x = x.permute(0, 2, 1) # (N, D, T/4) -> (N, T/4, D)
        x = F.silu(x)
        x = self.linear2(x)
        return x

def build_mlp_1024_to_256(hidden_size, projector_dim, z_dim):
    return MLP_1024_to_256(hidden_size, projector_dim, z_dim)

def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )