import numpy as np
import pandas as pd
import torch

# read static data
df_static = pd.read_csv('data/raw/kelmarsh/turbine_static.csv')

# fully connected graph
edge_index = np.array([[0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1,
                            2, 2, 2, 2, 2,
                            3, 3, 3, 3, 3,
                            4, 4, 4, 4, 4,
                            5, 5, 5, 5, 5],
                           [1, 2, 3, 4, 5,
                            0, 2, 3, 4, 5,
                            0, 1, 3, 4, 5,
                            0, 1, 2, 4, 5,
                            0, 1, 2, 3, 5,
                            0, 1, 2, 3, 4]])
np.save('data/processed/kelmarsh/edge_index.npy', edge_index)

# edge features: absolute latitudinal and longitudinal distances
# latitude at column index 9, longitude at 10
edge_attr_latlon = np.array([[abs(df_static.iloc[0,9] - df_static.iloc[1,9]),
                           abs(df_static.iloc[0,10] - df_static.iloc[1,10])],
                          [abs(df_static.iloc[0,9] - df_static.iloc[2,9]),
                           abs(df_static.iloc[0,10] - df_static.iloc[2,10])],
                          [abs(df_static.iloc[0,9] - df_static.iloc[3,9]),
                           abs(df_static.iloc[0,10] - df_static.iloc[3,10])],
                          [abs(df_static.iloc[0,9] - df_static.iloc[4,9]),
                           abs(df_static.iloc[0,10] - df_static.iloc[4,10])],
                          [abs(df_static.iloc[0,9] - df_static.iloc[5,9]),
                           abs(df_static.iloc[0,10] - df_static.iloc[5,10])],
                          [abs(df_static.iloc[1,9] - df_static.iloc[0,9]),
                           abs(df_static.iloc[1,10] - df_static.iloc[0,10])],
                          [abs(df_static.iloc[1,9] - df_static.iloc[2,9]),
                           abs(df_static.iloc[1,10] - df_static.iloc[2,10])],
                          [abs(df_static.iloc[1,9] - df_static.iloc[3,9]),
                           abs(df_static.iloc[1,10] - df_static.iloc[3,10])],
                          [abs(df_static.iloc[1,9] - df_static.iloc[4,9]),
                           abs(df_static.iloc[1,10] - df_static.iloc[4,10])],
                          [abs(df_static.iloc[1,9] - df_static.iloc[5,9]),
                           abs(df_static.iloc[1,10] - df_static.iloc[5,10])],
                          [abs(df_static.iloc[2,9] - df_static.iloc[0,9]),
                            abs(df_static.iloc[2,10] - df_static.iloc[0,10])],
                          [abs(df_static.iloc[2,9] - df_static.iloc[1,9]),
                           abs(df_static.iloc[2,10] - df_static.iloc[1,10])],
                          [abs(df_static.iloc[2,9] - df_static.iloc[3,9]),
                           abs(df_static.iloc[2,10] - df_static.iloc[3,10])],
                          [abs(df_static.iloc[2,9] - df_static.iloc[4,9]),
                           abs(df_static.iloc[2,10] - df_static.iloc[4,10])],
                          [abs(df_static.iloc[2,9] - df_static.iloc[5,9]),
                           abs(df_static.iloc[2,10] - df_static.iloc[5,10])],
                          [abs(df_static.iloc[3,9] - df_static.iloc[0,9]),
                           abs(df_static.iloc[3,10] - df_static.iloc[0,10])],
                          [abs(df_static.iloc[3,9] - df_static.iloc[1,9]),
                           abs(df_static.iloc[3,10] - df_static.iloc[1,10])],
                          [abs(df_static.iloc[3,9] - df_static.iloc[2,9]),
                           abs(df_static.iloc[3,10] - df_static.iloc[2,10])],
                          [abs(df_static.iloc[3,9] - df_static.iloc[4,9]),
                           abs(df_static.iloc[3,10] - df_static.iloc[4,10])],
                          [abs(df_static.iloc[3,9] - df_static.iloc[5,9]),
                           abs(df_static.iloc[3,10] - df_static.iloc[5,10])],
                          [abs(df_static.iloc[4,9] - df_static.iloc[0,9]),
                           abs(df_static.iloc[4,10] - df_static.iloc[0,10])],
                          [abs(df_static.iloc[4,9] - df_static.iloc[1,9]),
                           abs(df_static.iloc[4,10] - df_static.iloc[1,10])],
                          [abs(df_static.iloc[4,9] - df_static.iloc[2,9]),
                           abs(df_static.iloc[4,10] - df_static.iloc[2,10])],
                          [abs(df_static.iloc[4,9] - df_static.iloc[3,9]),
                           abs(df_static.iloc[4,10] - df_static.iloc[3,10])],
                          [abs(df_static.iloc[4,9] - df_static.iloc[5,9]),
                           abs(df_static.iloc[4,10] - df_static.iloc[5,10])],
                          [abs(df_static.iloc[5,9] - df_static.iloc[0,9]),
                           abs(df_static.iloc[5,10] - df_static.iloc[0,10])],
                          [abs(df_static.iloc[5,9] - df_static.iloc[1,9]),
                           abs(df_static.iloc[5,10] - df_static.iloc[1,10])],
                          [abs(df_static.iloc[5,9] - df_static.iloc[2,9]),
                           abs(df_static.iloc[5,10] - df_static.iloc[2,10])],
                          [abs(df_static.iloc[5,9] - df_static.iloc[3,9]),
                           abs(df_static.iloc[5,10] - df_static.iloc[3,10])],
                          [abs(df_static.iloc[5,9] - df_static.iloc[4,9]),
                           abs(df_static.iloc[5,10] - df_static.iloc[4,10])]],
                        dtype=torch.float)
#np.save('data/kelmarsh/kelmarsh_edge_attr.pt', edge_attr_latlon)

def compute_eucl_dist(i_1: int, i_2: int):
    dist = np.sqrt(abs(df_static.iloc[i_1,9] - df_static.iloc[i_2,9])**2 +
               abs(df_static.iloc[i_1,10]- df_static.iloc[i_2,10])**2)
    return dist

edge_attr_eucl = np.array([compute_eucl_dist(0,1),
                               compute_eucl_dist(0,2),
                               compute_eucl_dist(0,3),
                               compute_eucl_dist(0,4),
                               compute_eucl_dist(0,5),
                               compute_eucl_dist(1,0),
                               compute_eucl_dist(1,2),
                               compute_eucl_dist(1,3),
                               compute_eucl_dist(1,4),
                               compute_eucl_dist(1,5),
                               compute_eucl_dist(2,0),
                               compute_eucl_dist(2,1),
                               compute_eucl_dist(2,3),
                               compute_eucl_dist(2,4),
                               compute_eucl_dist(2,5),
                               compute_eucl_dist(3,0),
                               compute_eucl_dist(3,1),
                               compute_eucl_dist(3,2),
                               compute_eucl_dist(3,4),
                               compute_eucl_dist(3,5),
                               compute_eucl_dist(4,0),
                               compute_eucl_dist(4,1),
                               compute_eucl_dist(4,2),
                               compute_eucl_dist(4,3),
                               compute_eucl_dist(4,5),
                               compute_eucl_dist(5,0),
                               compute_eucl_dist(5,1),
                               compute_eucl_dist(5,2),
                               compute_eucl_dist(5,3),
                               compute_eucl_dist(5,4)])
np.save('data/processed/kelmarsh/edge_attr.npy', edge_attr_eucl)

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

df_filled = df.ffill()

n_timesteps = df.shape[0]
# node features; dim 1 = sites, dim 2 = # features, dim 3 = time
x = np.array([[df_filled.iloc[:,0], [df_static.iloc[0,7]]*n_timesteps],
              [df_filled.iloc[:,1], [df_static.iloc[1,7]]*n_timesteps],
              [df_filled.iloc[:,2], [df_static.iloc[2,7]]*n_timesteps],
              [df_filled.iloc[:,3], [df_static.iloc[3,7]]*n_timesteps],
              [df_filled.iloc[:,4], [df_static.iloc[4,7]]*n_timesteps],
              [df_filled.iloc[:,5], [df_static.iloc[5,7]]*n_timesteps]])
np.save('data/processed/kelmarsh/x.npy', x)

# node features; dim 1 = sites, dim 2 = # features, dim 3 = time
x = np.array([[1, 2, 3], [4, 5, 6]], np.float23)
