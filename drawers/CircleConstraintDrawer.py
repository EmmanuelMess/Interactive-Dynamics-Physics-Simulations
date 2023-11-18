import numpy as np
import pygame

from CircleConstraint import CircleConstraint
from drawers.Drawer import Drawer


class CircleConstraintDrawer(Drawer):
    def __init__(self, circleConstraint: CircleConstraint):
        self.circleConstraint = circleConstraint

    def draw(self, surface: pygame.Surface, origin: np.ndarray):
        # TODO fix sums
        c = (float((self.circleConstraint.center + origin)[0]), float((self.circleConstraint.center + origin)[1]))
        r = float(self.circleConstraint.radius)
        rect = pygame.Rect(c[0] - r, c[1] - r, r * 2, r * 2)
        pygame.draw.ellipse(surface, (0, 0, 0), rect, width=1)
