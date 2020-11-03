# --- default umap config -----
n_neighbors = 30
spread = 1.0
min_dist = 0.1
random_state = 42
verbose = False

# --- default stochastic embedding config -----
# fraction of samples used as neighbors
neighbor_frac = 0.7
# fraction of samples used as centroids
centroid_frac = 0.7
smoothing_epochs = 0
smoothing_neighbors = None
max_iter = 2000
# In this implementation, a is passed as a scaling 
# factor (s. embeddings.py - InvMapper.fit):
# scalar / embedding-range, this scalar is passed
# as the parameter.
a = 500 # / embedding range
# This corresponds to the parameter b of UMAP.
b = 1
