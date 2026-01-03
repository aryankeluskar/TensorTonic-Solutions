import numpy as np
from math import sqrt

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.linalg.norm(x - y))