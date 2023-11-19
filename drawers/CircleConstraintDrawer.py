import numpy as np
import pygame

from constraints.CircleConstraint import CircleConstraint
from drawers.Drawer import Drawer


class CircleConstraintDrawer(Drawer):
    def __init__(self, circleConstraint: CircleConstraint):
        self.circleConstraint = circleConstraint

    def draw(self, surface: pygame.Surface, origin: np.ndarray):
        c = self.circleConstraint.center + origin
        r = self.circleConstraint.radius
        rect = pygame.Rect(c[0] - r, c[1] - r, r * 2, r * 2)
        pygame.draw.ellipse(surface, (0, 0, 0), rect, width=1)
