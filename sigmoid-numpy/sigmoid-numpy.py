import numpy as np
from math import e

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x = np.asarray(x, dtype=float)
    return 1 / (1 + np.exp(-x))