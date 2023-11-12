from typing import List

import numpy as np
import pygame

from Constraint import Constraint
from Particle import Particle


class DistanceConstraint(Constraint):
    distance: np.float64

    def __init__(self, i: int, particleA: Particle, particleB: Particle, distance: np.float64):
        super().__init__(i, [particleA, particleB])
        self.distance = distance

    def C(self) -> np.float64:
        d = self.distance
        a, b = self.particles[0], self.particles[1]
        return np.sum((a.x-b.x)**2)/2 - np.sum(d**2)/2

    def dC(self) -> np.float64:
        a, b = self.particles[0], self.particles[1]
        return np.sum((a.x - b.x) * (a.v - b.v), dtype=np.float64)

    def d2C(self) -> np.float64:
        a, b = self.particles[0], self.particles[1]
        return np.sum((a.v - b.v) ** 2) + np.sum((a.x - b.x) * (a.a - b.a))

    def dCdq(self, x):
        return x

    def J(self) -> dict:
        return {
            self.particles[0].i: self.particles[0].x - self.particles[1].x,
            self.particles[1].i: self.particles[1].x - self.particles[0].x,
        }

    def dJ(self) -> dict:
        return { particle.i: np.array([1, 1], dtype=np.float64) for particle in self.particles }

    def surface(self, surface: pygame.Surface, origin: np.ndarray):
        a, b = self.particles[0].x + origin, self.particles[1].x + origin
        pygame.draw.line(surface, (0, 0, 0), (float(a[0]), float(a[1])), (float(b[0]), float(b[1])))