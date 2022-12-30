from tsp_solver.greedy_numpy import solve_tsp
import torch
import numpy as np
from functools import cache


@cache
def get_local_distance_minimizing_permutation(x):
    """Get a permutation that minimizes the travelling distance between the rows of x"""
    assert len(x.shape) == 2, "x should be interpretable as a list of vecs"
    n_vectors = x.shape[0]
    distances = (x[None, :, :] - x[:, None, :]).norm(dim=-1)
    distances = torch.cat((distances, torch.zeros(1, n_vectors)), axis=0)
    distances = torch.cat((distances, torch.zeros(n_vectors + 1, 1)), axis=1)
    tour = solve_tsp(distances.numpy())
    permutation = torch.tensor(
        tour[tour.index(n_vectors) + 1 :] + tour[: tour.index(n_vectors)]
    )
    return permutation


@cache
def get_seriation_permutations(x):
    perm_1, perm_2 = get_local_distance_minimizing_permutation(
        x
    ), get_local_distance_minimizing_permutation(x.T)
    return perm_1, perm_2


@cache
def seriate(x):
    perm_1, perm_2 = get_local_distance_minimizing_permutation(
        x
    ), get_local_distance_minimizing_permutation(x.T)
    return x[perm_1][:, perm_2], perm_1, perm_2
