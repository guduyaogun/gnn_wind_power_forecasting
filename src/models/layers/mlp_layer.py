import torch.nn.functional as F
from torch import nn


class MLPLayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout=0.05,
        activation="relu",
        norm_layer="layer",
    ):
        super().__init__()

        self.hidden = nn.Linear(input_size, output_size)
        if norm_layer == "layer":
            self.norm = nn.LayerNorm(output_size)
        elif norm_layer == "batch":
            self.norm = nn.BatchNorm1d(output_size)
        else:
            self.norm = None
        self.activation = F.relu if activation == "relu" else F.gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.hidden(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.dropout(self.activation(x))

        return x
