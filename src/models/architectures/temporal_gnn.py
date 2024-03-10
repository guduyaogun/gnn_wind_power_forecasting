import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN


class Model(torch.nn.Module):
    def __init__(
            self,
            num_static_node_features,
            num_timesteps_in,
            num_timesteps_out):
        super().__init__()
        self.tgnn = A3TGCN(in_channels=num_static_node_features,
                           out_channels=num_timesteps_in,
                           periods=num_timesteps_out)
        self.linear = torch.nn.Linear(num_timesteps_in, num_timesteps_out)

    def forward(self, x, edge_index, edge_weight):
        h = self.tgnn(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
