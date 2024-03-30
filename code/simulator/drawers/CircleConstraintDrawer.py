import numpy as np
import pygame

from simulator.constraints.CircleConstraint import CircleConstraint
from simulator.drawers.Drawer import Drawer


class CircleConstraintDrawer(Drawer):
    def __init__(self, circleConstraint: CircleConstraint) -> None:
        self.circleConstraint = circleConstraint

    def draw(self, surface: pygame.Surface, origin: np.ndarray) -> None:
        c = self.circleConstraint.center + origin
        r: float = self.circleConstraint.radius.item()
        rect = pygame.Rect((c[0] - r).item(), (c[1] - r).item(), r * 2, r * 2)
        pygame.draw.ellipse(surface, (0, 0, 0), rect, width=1)

    def getText(self) -> str:
        return ""
