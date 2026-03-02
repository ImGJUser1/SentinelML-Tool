import numpy as np

def ensure_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x

def l2_distance(a, b):
    return np.linalg.norm(a - b, axis=1)

def normalize(arr):
    arr = np.asarray(arr)
    if arr.std() == 0:
        return arr
    return (arr - arr.mean()) / (arr.std() + 1e-8)