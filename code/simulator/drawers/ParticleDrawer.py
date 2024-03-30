from typing import Final

import numpy as np
import pygame

from simulator.Particle import Particle
from simulator.drawers.Drawer import Drawer


class ParticleDrawer(Drawer):
    radius: Final[int] = 5
    fontSize: Final[int] = 10

    def __init__(self, particle: Particle) -> None:
        self.particle = particle
        self.font = pygame.font.SysFont("monospace", self.fontSize)
        self.label = self.font.render(f"{self.particle.index}", True, (0, 0, 0))

    def draw(self, surface: pygame.Surface, origin: np.ndarray) -> None:
        c = 10
        p = origin + self.particle.x
        aApplied = p + self.particle.aApplied * c
        aConstraint = p + self.particle.aConstraint * c
        pygame.draw.line(surface, (0, 255, 0), p, aApplied)
        pygame.draw.line(surface, (255, 0, 0), p, aConstraint)
        pygame.draw.circle(surface, (255, 0, 0) if self.particle.static else (0, 0, 255), p, self.radius)
        surface.blit(self.label, p + np.array([self.radius*0.5, self.radius*0.5]))

    def getText(self) -> str:
        return (f"p {self.particle.index}\n"
                f"  x {self.particle.x}\n"
                f"  v {self.particle.v}\n"
                f"  a {self.particle.a}")
