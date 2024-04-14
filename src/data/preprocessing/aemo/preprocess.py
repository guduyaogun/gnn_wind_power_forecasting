"""
This script is for preprocessing the raw data such that it can be used for GNN modeling.
It does not have to be run but is provided for transparency reasons only.
"""

import numpy as np
import pandas as pd

datapath_raw = "data/raw/aemo/"
datapath_processed = "data/processed/aemo/"

# read data
df = pd.read_csv(
    f"{datapath_raw}aemo_20222023.csv", index_col=0, parse_dates=True
).sort_index(axis=1)
df_loc = pd.read_excel(f"{datapath_raw}locations_wind_farms.xlsx")

# drop all columns with at least 1000 null values
null_counts = df.isnull().sum()
df.dropna(axis=1, thresh=len(df) - 1000, inplace=True)
print(f"{df.shape[1]} wind farms remain.")

# impute the remaining missing values with the corresponding previous value
df.fillna(method="ffill", inplace=True)

"""
We want to base our graph structure on spatial correlations. Computing correlations on the raw data
might give misleading results, as sometimes, individual wind farms produce 0MW because of maintenance,
for example, and this would not reflect actual spatial correlation. Therefore, we compute the pearson
correlation for each pair of wind farms, excluding time windows where at least one of the two wind farms
measured MW = 0.
"""

# Initialize lists to store highly correlated wind farm pairs
# for different correlation thresholds
geq05_startidx = []
geq05_endidx = []
geq06_startidx = []
geq06_endidx = []
geq07_startidx = []
geq07_endidx = []
geq08_startidx = []
geq08_endidx = []
geq09_startidx = []
geq09_endidx = []

# Iterate over unique pairs of columns
for i, col1 in enumerate(df.columns):
    for j, col2 in enumerate(df.columns):
        if i < j:
            # Exclude rows with zeros in one of the two columns
            df_temp = df[[col1, col2]]
            df_cleaned = df_temp[(df_temp != 0).all(axis=1)]
            # Compute correlation
            corr = df_cleaned[col1].corr(df_cleaned[col2], method="pearson")

            # add edges to lists (later used for edge_index)
            if corr >= 0.5:
                # edge from i to j
                geq05_startidx.append(i)
                geq05_endidx.append(j)
                # edge from j to i
                geq05_startidx.append(j)
                geq05_endidx.append(i)
            if corr >= 0.6:
                geq06_startidx.append(i)
                geq06_endidx.append(j)
                geq06_startidx.append(j)
                geq06_endidx.append(i)
            if corr >= 0.7:
                geq07_startidx.append(i)
                geq07_endidx.append(j)
                geq07_startidx.append(j)
                geq07_endidx.append(i)
            if corr >= 0.8:
                geq08_startidx.append(i)
                geq08_endidx.append(j)
                geq08_startidx.append(j)
                geq08_endidx.append(i)
            if corr >= 0.9:
                geq09_startidx.append(i)
                geq09_endidx.append(j)
                geq09_startidx.append(j)
                geq09_endidx.append(i)

# set edge_index lists
edge_idx_05 = np.array([geq05_startidx, geq05_endidx])
edge_idx_06 = np.array([geq06_startidx, geq06_endidx])
edge_idx_07 = np.array([geq07_startidx, geq07_endidx])
edge_idx_08 = np.array([geq08_startidx, geq08_endidx])
edge_idx_09 = np.array([geq09_startidx, geq09_endidx])

# np.save(f"{datapath_processed}edge_index_05.npy", edge_idx_05)
# np.save(f"{datapath_processed}edge_index_06.npy", edge_idx_06)
# np.save(f"{datapath_processed}edge_index_07.npy", edge_idx_07)
# np.save(f"{datapath_processed}edge_index_08.npy", edge_idx_08)
# np.save(f"{datapath_processed}edge_index_09.npy", edge_idx_09)

"""
We now want to define edge features: euclidean distances between wind farms.
"""

# select wind farms to match with df
select = [
    df_loc["Abbreviation"].to_list()[i] in df.columns for i in range(df_loc.shape[0])
]
df_loc = df_loc.iloc[select, :]


def compute_eucl_dist(i_1: int, i_2: int):
    dist = np.sqrt(
        abs(df_loc.iloc[i_1, 2] - df_loc.iloc[i_2, 2]) ** 2
        + abs(df_loc.iloc[i_1, 3] - df_loc.iloc[i_2, 3]) ** 2
    )
    return dist


edge_attr_eucl_05 = np.array(
    [
        compute_eucl_dist(edge_idx_05[0, i], edge_idx_05[1, i])
        for i in range(edge_idx_05.shape[1])
    ]
)
edge_attr_eucl_06 = np.array(
    [
        compute_eucl_dist(edge_idx_06[0, i], edge_idx_06[1, i])
        for i in range(edge_idx_06.shape[1])
    ]
)
edge_attr_eucl_07 = np.array(
    [
        compute_eucl_dist(edge_idx_07[0, i], edge_idx_07[1, i])
        for i in range(edge_idx_07.shape[1])
    ]
)
edge_attr_eucl_08 = np.array(
    [
        compute_eucl_dist(edge_idx_08[0, i], edge_idx_08[1, i])
        for i in range(edge_idx_08.shape[1])
    ]
)
edge_attr_eucl_09 = np.array(
    [
        compute_eucl_dist(edge_idx_09[0, i], edge_idx_09[1, i])
        for i in range(edge_idx_09.shape[1])
    ]
)

np.save(f"{datapath_processed}edge_attr_05.npy", edge_attr_eucl_05)
np.save(f"{datapath_processed}edge_attr_06.npy", edge_attr_eucl_06)
np.save(f"{datapath_processed}edge_attr_07.npy", edge_attr_eucl_07)
np.save(f"{datapath_processed}edge_attr_08.npy", edge_attr_eucl_08)
np.save(f"{datapath_processed}edge_attr_09.npy", edge_attr_eucl_09)

"""
Finally, we define the node features.
"""
x = np.array([df.iloc[:, i] for i in range(df.shape[1])])
np.save(f"{datapath_processed}x.npy", x)
