import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN


class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.num_static_node_features = args.num_static_node_features
        self.num_timesteps_in = args.num_timesteps_in
        self.num_timesteps_out = args.num_timesteps_out
        self.tgcn = TGCN(
            in_channels=self.num_timesteps_in,
            out_channels=self.num_timesteps_in,  # could be changed to other value
        )
        self.linear = torch.nn.Linear(self.num_timesteps_in, self.num_timesteps_out)

    def forward(self, x, edge_index, edge_weight, prev_hidden_state):
        h = self.tgcn(x, edge_index, edge_weight, prev_hidden_state)
        y = F.relu(h)
        y = self.linear(y)
        return y, h
