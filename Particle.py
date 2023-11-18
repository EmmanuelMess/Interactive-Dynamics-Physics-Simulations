import numpy as np

from drawers.Drawer import Drawer


class Particle:
    index: int
    x: np.ndarray
    v: np.ndarray
    a: np.ndarray
    aApplied: np.ndarray
    aConstraint: np.ndarray

    def __init__(self, i: int, x: np.ndarray, v: np.ndarray):
        self.index, self.x, self.v = i, x, v
        self.a, self.aApplied, self.aConstraint = np.zeros_like(v), np.zeros_like(v), np.zeros_like(v)
        self.drawer = None

    def initDrawer(self):
        from drawers.ParticleDrawer import ParticleDrawer # Prevent circular dependency
        self.drawer = ParticleDrawer(self)

    def getDrawer(self) -> Drawer:
        return self.drawer