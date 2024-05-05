import numpy as np
import torch
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class DataLoader:
    """
    Dataset loader for the AEMO wind farm data.
    """

    def __init__(self, args):
        super().__init__()
        self.features = []
        self.targets = []
        self.correlation_treshold = args.correlation_threshold
        self._read_data()

    def _read_data(self):

        edge_index = np.load(
            f"data/processed/aemo/edge_index_{self.correlation_treshold}.npy"
        )
        edge_attr = np.load(
            f"data/processed/aemo/edge_attr_{self.correlation_treshold}.npy"
        )
        X = np.load("data/processed/aemo/x.npy")

        self.edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Scale edge weights between 0 and 1
        edge_attr = (edge_attr - np.min(edge_attr)) / (
            np.max(edge_attr) - np.min(edge_attr)
        )
        self.edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Normalise X with z-score method
        means = np.mean(X, axis=1, keepdims=True)
        X = X - means
        stds = np.std(X, axis=1, keepdims=True)
        X = X / stds
        self.X = torch.tensor(X, dtype=torch.float)

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        """
        Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting the average wind speed using num_timesteps_in to predict the
        wind speed in the next num_timesteps_out

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[1] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, i + num_timesteps_in : j]).numpy())

        self.features = features
        self.targets = target

    def get_dataset(
        self, num_timesteps_in: int = 12, num_timesteps_out: int = 12
    ) -> StaticGraphTemporalSignal:
        """Returns data iterator for Kelmarsh dataset as an instance of the
        static graph temporal signal class.

        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Kelmarsh wind speed
                forecasting dataset.
        """
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(
            self.edge_index, self.edge_attr, self.features, self.targets
        )

        return dataset
