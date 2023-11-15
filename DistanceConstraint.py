import jax.numpy as jnp
from jax import jit, grad

import numpy as np
import pygame

from Constraint import Constraint
from Particle import Particle
from PositionApproximation import constructPositionFunction


class DistanceConstraint(Constraint):
    distance: np.float64

    def __init__(self, index: int, particleA: Particle, particleB: Particle, distance: np.float64):
        super().__init__(index, [particleA, particleB], DistanceConstraint.constraintTime)
        self.distance = distance

    @staticmethod
    @jit
    def constraint(distance, a: jnp.ndarray, b: jnp.ndarray):
        return jnp.sum((a - b) ** 2) / 2 - (distance ** 2) / 2

    @staticmethod
    @jit
    def constraintTime(t: jnp.float64, x: jnp.ndarray, params):
        a = x[0]
        b = x[1]
        positionApproximationA = constructPositionFunction(a[0], a[1], a[2])
        positionApproximationB = constructPositionFunction(b[0], b[1], b[2])
        return DistanceConstraint.constraint(params["distance"], positionApproximationA(t), positionApproximationB(t))

    def getArgs(self) -> dict:
        return {
            "distance": self.distance
        }

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