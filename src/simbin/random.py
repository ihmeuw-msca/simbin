"""
Random sampling function
"""
import numpy as np
from numpy import ndarray
from simbin.geometry import get_prob_vertices
from simbin.optim import adjust_prob_mat


def sample_simplex(ndim: int, size: int = 1) -> ndarray:
    # special case when n == 1
    if ndim == 1:
        return np.ones((size, ndim))

    points = np.hstack([np.zeros((size, 1)),
                        np.random.rand(size, ndim - 1),
                        np.ones((size, 1))])
    points.sort(axis=1)
    samples = np.diff(points)

    return samples


def sample_bin_prob(prob_mat: ndarray, size: int = 1) -> ndarray:
    vertices = get_prob_vertices(prob_mat)
    weights = sample_simplex(vertices.shape[0], size=size)
    return weights.dot(vertices)
