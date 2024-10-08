import numpy as np

from simulator.IndexedElement import IndexedElement
from simulator.drawers.Drawable import Drawable
from simulator.writers.Writable import Writable


class Particle(Drawable, Writable, IndexedElement):
    x: np.ndarray
    v: np.ndarray
    a: np.ndarray
    aApplied: np.ndarray
    aConstraint: np.ndarray
    static: bool

    def __init__(self, x: np.ndarray, static: bool = False) -> None:
        super().__init__()
        self.x = x
        self.static = static
        self.v = np.zeros_like(x)
        self.a = np.zeros_like(x)
        self.aApplied = np.zeros_like(x)
        self.aConstraint = np.zeros_like(x)

    def initDrawer(self) -> None:
        from simulator.drawers.ParticleDrawer import ParticleDrawer  # Prevent circular dependency
        self.setDrawer(ParticleDrawer(self))

    def initWriter(self) -> None:
        from simulator.writers.ParticleWriter import ParticleWriter  # Prevent circular dependency
        self.setWriter(ParticleWriter(self))
