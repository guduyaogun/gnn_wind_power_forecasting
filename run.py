import argparse
import random
import time

import numpy as np
import torch
from torch_geometric_temporal.signal import temporal_signal_split

import wandb
from src.data.dataloaders import aemo, kelmarsh
from src.models.architectures import mlp, temporal_gnn, tgcn


# TODO: Write tests for all parameter choices
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
        "--device",
        type=str,
        default="cpu",
        help="torch device, options: [cpu, cuda], default: cpu",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        help="model to use, options: [mlp, temporal_gnn, tgcn], default: mlp",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="number of epochs, default: 10",
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
        "--train_data_amount",
        type=int,
        default="10",
        help="percentage of training data to use for training, options: integer between 1 and 100, default: 10",
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
        "tgcn": tgcn,
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

    # TODO: Throw error if model == tgcn and num_static_node_features > 1 (not supported)

    train_data, test_data = temporal_signal_split(data, train_ratio=0.8)

    model = model_dict[args.model].Model(args)
    model.to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project_name,
            config={
                "model": args.model,
                "data": args.data,
                "num_timesteps_in": args.num_timesteps_in,
                "num_timesteps_out": args.num_timesteps_out,
            },
            name=f"{args.model}",
        )

    print(">>>>Start Training>>>>")
    start_time = time.time()
    for epoch in range(args.num_epochs):
        loss = 0
        step = 0
        hidden_state = None
        for snapshot in train_data[
            : train_data.snapshot_count * args.train_data_amount / 100
        ]:

            # Get right shape for node feature input
            if len(snapshot.x.shape) > 2 or args.model == "tgcn":
                x = snapshot.x
            else:
                x = snapshot.x.reshape(snapshot.x.shape[0], 1, snapshot.x.shape[1])

            x = x.to(device=args.device)
            edge_index = snapshot.edge_index.to(device=args.device)
            edge_attr = snapshot.edge_attr.to(device=args.device)
            y = snapshot.y.to(device=args.device)
            # Get model predictions
            if args.model == "tgcn":
                y_hat, hidden_state = model(x, edge_index, edge_attr, hidden_state)
            else:
                y_hat = model(x, edge_index, edge_attr)

            # Mean squared error
            loss = loss + torch.mean((y_hat.squeeze() - y) ** 2)
            step += 1

        loss = loss / (step + 1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.cpu().item()
        print(f"Epoch {epoch} train MSE: {loss:.4f}")
        if args.use_wandb:
            metrics = {"train_mse": loss}
            wandb.log(metrics)

        model.eval()
        loss = 0
        step = 0
        hidden_state = None
        horizon = 100

        for snapshot in test_data:

            # Get right shape for node feature input
            if len(snapshot.x.shape) > 2 or args.model == "tgcn":
                x = snapshot.x
            else:
                x = snapshot.x.reshape(snapshot.x.shape[0], 1, snapshot.x.shape[1])

            x = x.to(device=args.device)
            edge_index = snapshot.edge_index.to(device=args.device)
            edge_attr = snapshot.edge_attr.to(device=args.device)
            y = snapshot.y.to(device=args.device)
            # Get predictions
            if args.model == "tgcn":
                y_hat, hidden_state = model(x, edge_index, edge_attr, hidden_state)
            else:
                y_hat = model(x, edge_index, edge_attr)

            # Mean squared error
            loss = loss + torch.mean((y_hat.squeeze() - y) ** 2)

            step += 1
            if step > horizon:
                break

        loss = loss / (step + 1)
        loss = loss.cpu().item()
        print(f"Epoch {epoch} test MSE: {loss:.4f}")
        if args.use_wandb:
            metrics = {"test_mse": loss}
            wandb.log(metrics)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time}")
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
