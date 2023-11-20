import numpy as np

from drawers.Drawable import Drawable
from drawers.Drawer import Drawer


class Particle(Drawable):
    index: int
    x: np.ndarray
    v: np.ndarray
    a: np.ndarray
    aApplied: np.ndarray
    aConstraint: np.ndarray

    def __init__(self, i: int, x: np.ndarray):
        super().__init__()
        self.index, self.x = i, x
        self.v, self.a, self.aApplied, self.aConstraint = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        self.drawer = None

    def initDrawer(self):
        from drawers.ParticleDrawer import ParticleDrawer # Prevent circular dependency
        self.setDrawer(ParticleDrawer(self))
