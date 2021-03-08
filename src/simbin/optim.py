"""
Optimization to adjust the constraints
"""
from typing import Dict
import numpy as np
from numpy import ndarray
from scipy.optimize import LinearConstraint, minimize


def adjust_prob_mat(prob_mat: ndarray,
                    weights: ndarray = None,
                    x0: ndarray = None,
                    rtol: float = 1e-4,
                    atol: float = 1e-4,
                    options: Dict = None) -> ndarray:
    ndim = prob_mat.shape[0]
    size = 2**ndim

    weights = np.ones((ndim, ndim)) if weights is None else weights
    A = np.empty(shape=(0, size))
    b = np.empty(0)
    w = np.empty(0)

    ind = [[0, 1]]*ndim
    for i in range(ndim):
        for j in range(i, ndim):
            ind[i] = [1]
            ind[j] = [1]
            row = np.zeros((2,)*ndim)
            row[tuple(np.meshgrid(*ind))] = 1.0
            A = np.vstack([A, row.ravel()])
            b = np.hstack([b, prob_mat[i, j]])
            w = np.hstack([w, weights[i, j]])
            ind[i] = [0, 1]
            ind[j] = [0, 1]

    # create optimization problem
    def objective(x: ndarray) -> float:
        return 0.5*w.dot((A.dot(x) - b)**2)

    def gradient(x: ndarray) -> ndarray:
        return (A.T*w).dot(A.dot(x) - b)

    def hessian(x: ndarray) -> ndarray:
        return (A.T*w).dot(A)

    if x0 is None:
        x0 = np.ones(size)/size
    bounds = np.hstack([np.zeros((size, 1)), np.ones((size, 1))])
    constraints = [LinearConstraint(np.ones((1, size)), np.ones(1), np.ones(1))]

    result = minimize(
        objective, x0,
        method="trust-constr",
        jac=gradient,
        hess=hessian,
        constraints=constraints,
        bounds=bounds,
        options=options
    )

    x = result.x
    adjusted_b = A.dot(x)
    adjusted_prob_mat = prob_mat.copy()
    if not np.allclose(adjusted_b, b, rtol=rtol, atol=atol):
        print("Adjust probablility matrix.")
        k = 0
        for i in range(ndim):
            for j in range(i, ndim):
                adjusted_prob_mat[i, j] = adjusted_b[k]
                k += 1
    return adjusted_prob_mat, x
