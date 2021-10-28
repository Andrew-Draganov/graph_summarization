import numpy as np

def discard_unlikely(vv, D, eps=0.001):
    degree_vector = np.diag(D)
    discard_indices = np.where(vv < D * eps)
    vv[discard_indices, 0] = 0
    return vv

def symmetrize(mat, n):
    # make the matrix symmetric
    i_upper = np.triu_indices(n, -1)
    mat[i_upper] = mat.T[i_upper]
    return mat

# def determine_subgraph(walkers, adj_mat, edge_types, K):
