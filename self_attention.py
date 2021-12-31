import numpy as np


def apply_self_attention(x):
    weight_matrix = np.apply_along_axis(
        func1d=lambda x: np.exp(x) / sum(np.exp(x)),
        axis=0,
        arr=np.dot(x, x.T),
    )
    return np.dot(weight_matrix, x)
