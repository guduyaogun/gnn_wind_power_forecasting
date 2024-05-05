import argparse
import random

import numpy as np
import torch
from torch_geometric_temporal.signal import temporal_signal_split

import wandb
from src.data.dataloaders import aemo, kelmarsh
from src.models.architectures import mlp, temporal_gnn


def main():
    fix_seed = 42
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="Wind Power Forecasting")
    parser.add_argument(
        "--correlation_threshold",
        type=str,
        default="08",
        help="correlation threshold of when to connect wind sites with an edge in the input graph, options: [05, 06, 07, 08, 09], default: 08",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="kelmarsh",
        help="dataset to use, options: [aemo, kelmarsh], default: kelmarsh",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        help="model to use, options: [mlp, temporal_gnn], default: mlp",
    )
    parser.add_argument(
        "--num_timesteps_in",
        type=int,
        default=12,
        help="length (number of consecutive data points) of the look back window, default: 12",
    )
    parser.add_argument(
        "--num_timesteps_out",
        type=int,
        default=12,
        help="number of consecutive data points to predict, default: 12",
    )
    parser.add_argument(
        "--use_wandb",
        type=bool,
        default=False,
        help="whether to track experiment in Weights & Biases, see https://wandb.ai/site, default: False",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="sulphur-crested-cockatoo",
        help="name of wandb project to initialize, default: sulphur-crested-cockatoo",
    )
    args = parser.parse_args()

    model_dict = {
        "mlp": mlp,
        "temporal_gnn": temporal_gnn,
    }

    data_dict = {"kelmarsh": kelmarsh, "aemo": aemo}

    dataloader = data_dict[args.data].DataLoader(args)
    data = dataloader.get_dataset(
        num_timesteps_in=args.num_timesteps_in, num_timesteps_out=args.num_timesteps_out
    )

    if len(data[0].x.shape) > 2:
        args.num_static_node_features = data[0].x.shape[1]
    else:
        args.num_static_node_features = 1

    train_data, test_data = temporal_signal_split(data, train_ratio=0.8)

    model = model_dict[args.model].Model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    if args.usewandb:
        wandb.init(
            project="wandb-test-2",
            config={
                "model": args.model,
                "data": args.data,
                "num_timesteps_in": args.num_timesteps_in,
                "num_timesteps_out": args.num_timesteps_out,
            },
            name=f"{args.model}",
        )

    print(">>>>Start Training>>>>")
    for epoch in range(10):
        loss = 0
        step = 0
        for snapshot in train_data[:1000]:
            # Get model predictions
            if len(snapshot.x.shape) > 2:
                y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            else:
                y_hat = model(
                    snapshot.x.reshape(snapshot.x.shape[0], 1, snapshot.x.shape[1]),
                    snapshot.edge_index,
                    snapshot.edge_attr,
                )
            # Mean squared error
            loss = loss + torch.mean((y_hat.squeeze() - snapshot.y) ** 2)
            step += 1

        loss = loss / (step + 1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch} train MSE: {loss.item():.4f}")
        if args.usewandb:
            metrics = {"train_mse": loss.item()}
            wandb.log(metrics)

        model.eval()
        loss = 0
        step = 0
        horizon = 100

        for snapshot in test_data:
            # Get predictions
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            # Mean squared error
            loss = loss + torch.mean((y_hat.squeeze() - snapshot.y) ** 2)

            step += 1
            if step > horizon:
                break

        loss = loss / (step + 1)
        loss = loss.item()
        print(f"Epoch {epoch} test MSE: {loss:.4f}")
        if args.usewandb:
            metrics = {"test_mse": loss}
            wandb.log(metrics)

    if args.usewandb:
        wandb.finish()

    model.eval()
    loss = 0
    step = 0
    horizon = 100

    print(">>>>Testing>>>>")
    for snapshot in test_data:
        # Get predictions
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        # Mean squared error
        loss = loss + torch.mean((y_hat.squeeze() - snapshot.y) ** 2)

        step += 1
        if step > horizon:
            break

    loss = loss / (step + 1)
    loss = loss.item()
    print(f"Test MSE: {loss:.4f}")


if __name__ == "__main__":
    main()
