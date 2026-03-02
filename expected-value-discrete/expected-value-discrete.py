import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """

    if not sum(p) == 1:
        raise ValueError

    out = 0
    for i in range(len(x)):
        out += x[i] * p[i]

    return out