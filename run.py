import argparse
import random

import numpy as np
import torch
from torch_geometric_temporal.signal import temporal_signal_split

from src.data.dataloaders import kelmarsh
from src.models.architectures import mlp, temporal_gnn


def main():
    fix_seed = 42
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="Wind Power Forecasting")
    parser.add_argument(
        "--model",
        type=str,
        default="TemporalGNN",
        help="model name, options: [MLP, TemporalGNN]",
    )
    parser.add_argument(
        "--data", type=str, default="kelmarsh", help="dataset name, options: [kelmarsh]"
    )
    parser.add_argument(
        "--num_timesteps_in",
        type=int,
        default=12,
        help="length (number of consecutive data points) of the look back window",
    )
    parser.add_argument(
        "--num_timesteps_out",
        type=int,
        default=12,
        help="number of consecutive data points to predict",
    )
    args = parser.parse_args()

    model_dict = {
        "MLP": mlp,
        "TemporalGNN": temporal_gnn,
    }

    data_dict = {
        "kelmarsh": kelmarsh,
    }

    dataloader = data_dict[args.data].DataLoader()
    data = dataloader.get_dataset(
        num_timesteps_in=args.num_timesteps_in, num_timesteps_out=args.num_timesteps_out
    )
    args.num_static_node_features = data[0].x.shape[1]
    train_data, test_data = temporal_signal_split(data, train_ratio=0.8)

    model = model_dict[args.model].Model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    print(">>>>Start Training>>>>")
    for epoch in range(10):
        loss = 0
        step = 0
        for snapshot in train_data[:1000]:
            # Get model predictions
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            # Mean squared error
            loss = loss + torch.mean((y_hat.squeeze() - snapshot.y) ** 2)
            step += 1

        loss = loss / (step + 1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch} train MSE: {loss.item():.4f}")

    model.eval()
    loss = 0
    step = 0
    horizon = 100

    predictions = []
    labels = []

    print(">>>>Testing>>>>")
    for snapshot in test_data:
        # Get predictions
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        # Mean squared error
        loss = loss + torch.mean((y_hat.squeeze() - snapshot.y) ** 2)
        # Store for analysis below
        predictions.append(y_hat)
        labels.append(snapshot.y)

        step += 1
        if step > horizon:
            break

    loss = loss / (step + 1)
    loss = loss.item()
    print(f"Test MSE: {loss:.4f}")


if __name__ == "__main__":
    main()
