"""
This script is for preprocessing the raw data such that it can be used for GNN modeling.
It does not have to be run but is provided for transparency reasons only.
We also provide an alternative for edge features (latitudinal and longitudinal distances),
but it has not been used so far.
"""

import os
import shutil
import zipfile

import numpy as np
import pandas as pd

"""
We first adress the extracting and merging of the several ZIP folders one obtains when
downloading the data.
"""

datapath_raw = "data/raw/aemo/"
datapath_processed = "data/processed/aemo/"

# we first extract the downloaded ZIP files
for i in range(20231101, 20231131):
    with zipfile.ZipFile(f"{datapath_raw}PUBLIC_DISPATCHSCADA_{i}.zip", "r") as zip_ref:
        zip_ref.extractall(f"{datapath_raw}aemo_2023_{i}")

for i in range(20231201, 20231232):
    with zipfile.ZipFile(f"{datapath_raw}PUBLIC_DISPATCHSCADA_{i}.zip", "r") as zip_ref:
        zip_ref.extractall(f"{datapath_raw}aemo_2023_{i}")


# We then extract the ZIP files, which where in the original ZIP files
def extract_zip_files(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Iterate over each file
    for file_name in files:
        # Check if the file is a zip file
        if file_name.endswith(".zip"):
            file_path = os.path.join(folder_path, file_name)
            # Open the zip file
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                # Extract all contents to the folder
                zip_ref.extractall(folder_path)
            print(f"Extracted {file_name}")


# Call the function to extract zip files
for i in range(20231101, 20231131):
    extract_zip_files(f"{datapath_raw}aemo_2023_{i}")

for i in range(20231201, 20231232):
    extract_zip_files(f"{datapath_raw}aemo_2023_{i}")

# Delete all ZIP files within the new folders
for i in range(20231101, 20231131):
    files = os.listdir(f"{datapath_raw}aemo_2023_{i}")
    for file_name in files:
        # Check if the file is a zip file
        if file_name.endswith(".zip"):
            file_path = os.path.join(f"{datapath_raw}aemo_2023_{i}", file_name)
            print(file_path)
            os.remove(file_path)

for i in range(20231201, 20231232):
    files = os.listdir(f"{datapath_raw}aemo_2023_{i}")
    for file_name in files:
        # Check if the file is a zip file
        if file_name.endswith(".zip"):
            file_path = os.path.join(f"{datapath_raw}aemo_2023_{i}", file_name)
            os.remove(file_path)

# Join all files within the first folder
start = 20231101
files = os.listdir(f"{datapath_raw}aemo_2023_{start}")
file_name0 = files[0]
file_path = f"{datapath_raw}aemo_2023_{start}/{file_name0}"
df = pd.read_csv(file_path, skiprows=[0], usecols=[4, 5, 6])
df_t = df.pivot(index="SETTLEMENTDATE", columns="DUID", values="SCADAVALUE").iloc[
    1:, 1:
]
for file_name in files[1:]:
    file_path = f"{datapath_raw}aemo_2023_{start}/{file_name}"
    df_new = pd.read_csv(file_path, skiprows=[0], usecols=[4, 5, 6])
    df_t_new = df_new.pivot(
        index="SETTLEMENTDATE", columns="DUID", values="SCADAVALUE"
    ).iloc[1:, 1:]
    df_t = pd.concat([df_t, df_t_new])

# Join the files in the upcoming folders underneath
for i in range(start + 1, 20231131):
    files = os.listdir(f"{datapath_raw}aemo_2023_{i}")
    print(f"Connect folder {i}")
    for file_name in files:
        file_path = f"{datapath_raw}aemo_2023_{i}/{file_name}"
        df_new = pd.read_csv(file_path, skiprows=[0], usecols=[4, 5, 6])
        df_t_new = df_new.pivot(
            index="SETTLEMENTDATE", columns="DUID", values="SCADAVALUE"
        ).iloc[1:, 1:]
        df_t = pd.concat([df_t, df_t_new])

# df_t.to_csv(f"{datapath_raw}aemo_2023_11.csv")

for i in range(20231201, 20231232):
    files = os.listdir(f"{datapath_raw}aemo_2023_{i}")
    print(f"Connect folder {i}")
    for file_name in files:
        file_path = f"{datapath_raw}aemo_2023_{i}/{file_name}"
        df_new = pd.read_csv(file_path, skiprows=[0], usecols=[4, 5, 6])
        df_t_new = df_new.pivot(
            index="SETTLEMENTDATE", columns="DUID", values="SCADAVALUE"
        ).iloc[1:, 1:]
        df_t = pd.concat([df_t, df_t_new])

df_t.to_csv(f"{datapath_raw}aemo_2023_1112.csv")

# Delete all folders
for i in range(20231101, 20231131):
    folder_path = f"{datapath_raw}aemo_2023_{i}"
    shutil.rmtree(folder_path)

for i in range(20231201, 20231232):
    folder_path = f"{datapath_raw}aemo_2023_{i}"
    shutil.rmtree(folder_path)


# read data
df = pd.read_csv(f"{datapath_raw}aemo_20222023.csv", index_col=0, parse_dates=True)
df_loc = pd.read_excel(f"{datapath_raw}locations_wind_farms.xlsx")

"""
We want to base our graph structure on spatial correlations. Computing correlations on the raw data
might give misleading results, as sometimes, individual wind farms produce 0MW because of maintenance,
for example, and this would not reflect actual spatial correlation. Therefore, we compute the pearson
correlation for each pair of wind farms, excluding time windows where at least one of the two wind farms
measured MW = 0. Moreover, we only take time intervals into account where measurements exist.
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
            # Exclude rows with missing values or zeros in one of the two columns
            df_temp = df[[col1, col2]].dropna()
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

# reorder locations to match with df
df_loc_filtered = df_loc[df_loc["Abbreviation"].isin(df.columns)]
df_loc_filtered_indexed = df_loc_filtered.set_index("Abbreviation")
df_loc_reordered = df_loc_filtered_indexed.loc[df.columns, :].reset_index()


def compute_eucl_dist(i_1: int, i_2: int):
    dist = np.sqrt(
        abs(df_loc_reordered.iloc[i_1, 2] - df_loc_reordered.iloc[i_2, 2]) ** 2
        + abs(df_loc_reordered.iloc[i_1, 3] - df_loc_reordered.iloc[i_2, 3]) ** 2
    )
    return dist


edge_attr_eucl = np.array(
    [
        compute_eucl_dist(i, j)
        for i in range(df_loc_reordered.shape[0])
        for j in range(df_loc_reordered.shape[0])
        if i != j
    ]
)

# np.save(f'{datapath_processed}edge_attr.npy', edge_attr_eucl)

"""
Finally, we define the node features.
"""
x = np.array(
    [
        [df_filled.iloc[:, 0], [df_static.iloc[0, 7]] * n_timesteps],
        [df_filled.iloc[:, 1], [df_static.iloc[1, 7]] * n_timesteps],
        [df_filled.iloc[:, 2], [df_static.iloc[2, 7]] * n_timesteps],
        [df_filled.iloc[:, 3], [df_static.iloc[3, 7]] * n_timesteps],
        [df_filled.iloc[:, 4], [df_static.iloc[4, 7]] * n_timesteps],
        [df_filled.iloc[:, 5], [df_static.iloc[5, 7]] * n_timesteps],
    ]
)
# np.save('data/processed/kelmarsh/x.npy', x)
