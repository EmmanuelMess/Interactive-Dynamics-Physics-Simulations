import numpy as np
import pygame

from DistanceConstraint import DistanceConstraint
from drawers.Drawer import Drawer


class DistanceConstraintDrawer(Drawer):
    def __init__(self, distanceConstraint: DistanceConstraint):
        self.distanceConstraint = distanceConstraint

    def draw(self, surface: pygame.Surface, origin: np.ndarray):
        a, b = self.distanceConstraint.particles[0].x + origin, self.distanceConstraint.particles[1].x + origin
        pygame.draw.line(surface, (0, 0, 0), a, b)