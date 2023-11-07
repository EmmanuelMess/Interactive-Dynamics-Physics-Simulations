import numpy as np


class Particle:
    i: int
    x: np.ndarray
    v: np.ndarray
    a: np.ndarray
    aApplied: np.ndarray
    aConstraint: np.ndarray

    def __init__(self, i: int, x: np.ndarray, v: np.ndarray):
        self.i, self.x, self.v = i, x, v
        self.a, self.aApplied, self.aConstraint = np.zeros_like(v), np.zeros_like(v), np.zeros_like(v)