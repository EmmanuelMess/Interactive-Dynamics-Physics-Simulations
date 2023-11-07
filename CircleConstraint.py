from typing import List

import numpy as np
import pygame

from Constraint import Constraint
from Particle import Particle


class CircleConstraint(Constraint):
    radius: np.float64

    def __init__(self,  i: int, particle: Particle, center: np.ndarray, radius: np.float64):
        super().__init__(i, [particle])
        self.center, self.radius = center, radius

    def C(self) -> np.float64:
        r = self.radius
        p = self.particles[0]
        return (np.sum((p.x - self.center)**2))/2 - (r**2)/2

    def dC(self) -> np.float64:
        p = self.particles[0]
        return np.sum((p.x - self.center) * p.v)

    def d2C(self) -> np.float64:
        p = self.particles[0]
        return np.sum(p.v ** 2) + np.sum((p.x - self.center) * p.a)

    def dCdq(self, x):
        return x - self.center

    def d2Cdq(self):
        return np.array([1, 1], dtype=np.float64)

    def J(self) -> dict:
        return { particle.i: self.dCdq(particle.x) for particle in self.particles }

    def dJ(self) -> dict:
        return { particle.i: self.d2Cdq() for particle in self.particles }

    def surface(self, surface: pygame.Surface, origin: np.ndarray):
        c = (float((self.center+origin)[0]), float((self.center+origin)[1]))
        r = float(self.radius)
        pygame.draw.circle(surface, (0, 0, 0), c, r)
        pygame.draw.circle(surface, (255, 255, 255, 0), c, r - 1)