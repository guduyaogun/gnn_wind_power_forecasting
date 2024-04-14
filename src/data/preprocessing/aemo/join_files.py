"""
This script is for preprocessing the raw data such that it can be used for GNN modeling.
It does not have to be run but is provided for transparency reasons only.
"""

import os
import shutil
import zipfile

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
