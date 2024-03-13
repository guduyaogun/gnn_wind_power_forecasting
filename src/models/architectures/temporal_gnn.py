import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN


class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_static_node_features = args.num_static_node_features
        self.num_timesteps_in = args.num_timesteps_in
        self.num_timesteps_out = args.num_timesteps_out
        self.tgnn = A3TGCN(in_channels=self.num_static_node_features,
                           out_channels=self.num_timesteps_in,
                           periods=self.num_timesteps_out)
        self.linear = torch.nn.Linear(self.num_timesteps_in, self.num_timesteps_out)

    def forward(self, x, edge_index, edge_weight):
        h = self.tgnn(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
