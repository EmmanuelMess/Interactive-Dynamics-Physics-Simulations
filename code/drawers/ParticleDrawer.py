from typing import Final

import numpy as np
import pygame

from Particle import Particle
from drawers.Drawer import Drawer


class ParticleDrawer(Drawer):
    radius: Final[int] = 5
    fontSize: Final[int] = 10

    def __init__(self, particle: Particle):
        self.particle = particle
        self.font = pygame.font.SysFont("monospace", self.fontSize)
        self.label = self.font.render(f"{self.particle.index}", True, (0, 0, 0))

    def draw(self, surface: pygame.Surface, origin: np.ndarray):
        c = 100
        p = origin + self.particle.x
        a = p + self.particle.a * c
        pygame.draw.line(surface, (255, 0, 0), p, a)
        pygame.draw.circle(surface, (0, 0, 255), p, self.radius)
        surface.blit(self.label, p + np.array([self.radius*0.5, self.radius*0.5]))

    def getText(self) -> str:
        return (f"p {self.particle.index}\n"
                f"  x {self.particle.x}\n"
                f"  v {self.particle.v}\n"
                f"  a {self.particle.a}")
