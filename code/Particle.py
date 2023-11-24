import numpy as np

from IndexedElement import IndexedElement
from drawers.Drawable import Drawable


class Particle(Drawable, IndexedElement):
    x: np.ndarray
    v: np.ndarray
    a: np.ndarray
    aApplied: np.ndarray
    aConstraint: np.ndarray
    static: bool

    def __init__(self, x: np.ndarray, static: bool = False):
        super().__init__()
        self.x = x
        self.static = static
        self.v = np.zeros_like(x)
        self.a = np.zeros_like(x)
        self.aApplied = np.zeros_like(x)
        self.aConstraint = np.zeros_like(x)

    def initDrawer(self):
        from drawers.ParticleDrawer import ParticleDrawer # Prevent circular dependency
        self.setDrawer(ParticleDrawer(self))
