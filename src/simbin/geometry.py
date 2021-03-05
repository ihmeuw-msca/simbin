"""
Geometry for probability cube
"""
import numpy as np
import cdd
from numpy import ndarray


def get_prob_vertices(prob_mat: ndarray) -> ndarray:
    ndim = prob_mat.shape[0]
    size = 2**ndim

    # generate box constaints
    I = np.identity(size)
    A = np.vstack([I, -I])
    b = np.hstack([np.ones(size), np.zeros(size)])

    # sum of the probablity equal to 1
    A = np.vstack([A, np.ones(size)])
    b = np.hstack([b, 1.0])

    # equality constraints from probility matrix
    ind = [[0, 1]]*ndim

    for i in range(ndim):
        for j in range(i, ndim):
            ind[i] = [1]
            ind[j] = [1]
            row = np.zeros((2,)*ndim)
            row[tuple(np.meshgrid(*ind))] = 1.0
            A = np.vstack([A, row.ravel()])
            b = np.hstack([b, prob_mat[i, j]])
            ind[i] = [0, 1]
            ind[j] = [0, 1]

    # convert to cddlib
    mat = np.insert(-A, 0, b, axis=1)
    mat = cdd.Matrix(mat)
    mat.rep_type = cdd.RepType.INEQUALITY
    mat.lin_set = frozenset(range(2*size, A.shape[0]))

    poly = cdd.Polyhedron(mat)
    vertices_and_rays = np.array(poly.get_generators())
    if vertices_and_rays.size == 0:
        raise ValueError("No feasible probablility.")

    vertices = vertices_and_rays[:, 1:]

    return vertices
