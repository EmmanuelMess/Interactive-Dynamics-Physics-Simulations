import numpy as np
import pygame

from simulator.constraints.DistanceConstraint import DistanceConstraint
from simulator.drawers.Drawer import Drawer


class DistanceConstraintDrawer(Drawer):
    def __init__(self, distanceConstraint: DistanceConstraint):
        self.distanceConstraint = distanceConstraint

    def draw(self, surface: pygame.Surface, origin: np.ndarray) -> None:
        a = origin + self.distanceConstraint.particles[0].x * np.array([1, -1], dtype=np.float64)
        b = origin + self.distanceConstraint.particles[1].x * np.array([1, -1], dtype=np.float64)
        pygame.draw.line(surface, (0, 0, 0), (a[0].item(), a[1].item()), (b[0].item(), b[1].item()))

    def getText(self) -> str:
        return ""
