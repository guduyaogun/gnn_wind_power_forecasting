"""
This script is for preprocessing the raw data such that it can be used for GNN modeling.
It does not have to be run but is provided for transparency reasons only.
We also provide an alternative for edge features (latitudinal and longitudinal distances),
but it has not been used so far.
"""

import numpy as np
import pandas as pd

# read static data
df_static = pd.read_csv('data/raw/kelmarsh/turbine_static.csv')
n_nodes = df_static.shape[0]

# define adjacency (here: fully connected graph)
node_idx = list(range(n_nodes))
edge_idx = np.array([[i for i in range(6) for _ in range(5)],
                     [x for j in range(6) for x in (node_idx[:j] + node_idx[j+1:])]
                    ])
#np.save('data/processed/kelmarsh/edge_index.npy', edge_idx)

# define edge features: absolute latitudinal and longitudinal distances
# latitude at column index 9, longitude at 10
edge_attr_latlon = np.array([[abs(df_static.iloc[i,9] - df_static.iloc[j,9]),
                              abs(df_static.iloc[i,10] - df_static.iloc[j,10])]
                              for i in range(6) for j in range(6) if i != j])
#np.save('data/kelmarsh/kelmarsh_edge_attr.pt', edge_attr_latlon)


# define 1d alternative (which we currently use): euclidean distance
def compute_eucl_dist(i_1: int, i_2: int):
    dist = np.sqrt(abs(df_static.iloc[i_1,9] - df_static.iloc[i_2,9])**2 +
               abs(df_static.iloc[i_1,10]- df_static.iloc[i_2,10])**2)
    return dist

edge_attr_eucl = np.array([compute_eucl_dist(i,j)
                           for i in range(6) for j in range(6) if i != j])
#np.save('data/processed/kelmarsh/edge_attr.npy', edge_attr_eucl)

# read turbine data
t1 = pd.read_csv('data/raw/kelmarsh/turbine1.csv', skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                 index_col=0, parse_dates=True)
t2 = pd.read_csv('data/raw/kelmarsh/turbine2.csv', skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                 index_col=0, parse_dates=True)
t3 = pd.read_csv('data/raw/kelmarsh/turbine3.csv', skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                 index_col=0, parse_dates=True)
t4 = pd.read_csv('data/raw/kelmarsh/turbine4.csv', skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                 index_col=0, parse_dates=True)
t5 = pd.read_csv('data/raw/kelmarsh/turbine5.csv', skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                 index_col=0, parse_dates=True)
t6 = pd.read_csv('data/raw/kelmarsh/turbine6.csv', skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                 index_col=0, parse_dates=True)

df = pd.DataFrame({'Turbine 1': t1['Wind speed (m/s)'],
                       'Turbine 2': t2['Wind speed (m/s)'],
                       'Turbine 3': t3['Wind speed (m/s)'],
                       'Turbine 4': t4['Wind speed (m/s)'],
                       'Turbine 5': t5['Wind speed (m/s)'],
                       'Turbine 6': t6['Wind speed (m/s)']})

# fill the very few NAs (see eda notebook) with the previous value
df_filled = df.ffill()

n_timesteps = df.shape[0]
# node features; dim 1 = sites, dim 2 = # features, dim 3 = time
# here we use as features the wind speed over time and the turbine hub height
x = np.array([[df_filled.iloc[:,0], [df_static.iloc[0,7]]*n_timesteps],
              [df_filled.iloc[:,1], [df_static.iloc[1,7]]*n_timesteps],
              [df_filled.iloc[:,2], [df_static.iloc[2,7]]*n_timesteps],
              [df_filled.iloc[:,3], [df_static.iloc[3,7]]*n_timesteps],
              [df_filled.iloc[:,4], [df_static.iloc[4,7]]*n_timesteps],
              [df_filled.iloc[:,5], [df_static.iloc[5,7]]*n_timesteps]])
#np.save('data/processed/kelmarsh/x.npy', x)
