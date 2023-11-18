import numpy as np
import pygame

from Particle import Particle
from drawers.Drawer import Drawer


class ParticleDrawer(Drawer):
    def __init__(self, particle: Particle):
        self.particle = particle
        self.font = pygame.font.SysFont("monospace", 11)
        self.label = self.font.render(f"{self.particle.i}", 1, (0, 0, 0))

    def draw(self, surface: pygame.Surface, origin: np.ndarray):
        c = 100
        p = origin + self.particle.x
        a = p + self.particle.a * c
        aApplied = p + self.particle.aApplied * c
        aConstraint = p + self.particle.aConstraint * c
        pygame.draw.line(surface, (0, 0, 0), p, a)
        pygame.draw.line(surface, (0, 255, 0), p, aApplied)
        pygame.draw.line(surface, (255, 0, 0), p, aConstraint)
        pygame.draw.circle(surface, (0, 0, 255), p, 5)
        surface.blit(self.label, p)
