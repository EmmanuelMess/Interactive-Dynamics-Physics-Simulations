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
        super().__init__(index, [particleA, particleB], DistanceConstraint.constraint,
                         DistanceConstraint.constraintTime)
        self.distance = distance

    @staticmethod
    @jit
    def constraint(x: jnp.ndarray, params: dict):
        aPosition = x[0]
        bPosition = x[1]
        return jnp.sum((aPosition - bPosition) ** 2) / 2 - (params["distance"] ** 2) / 2

    @staticmethod
    @jit
    def constraintTime(t: jnp.float64, x: jnp.ndarray, params: dict):
        a = x[0]
        b = x[1]
        positionApproximationA = constructPositionFunction(a[0], a[1], a[2])
        positionApproximationB = constructPositionFunction(b[0], b[1], b[2])
        return DistanceConstraint.constraint(jnp.array([positionApproximationA(t), positionApproximationB(t)]), params)

    def getArgs(self) -> dict:
        return {
            "distance": self.distance
        }

    def surface(self, surface: pygame.Surface, origin: np.ndarray):
        a, b = self.particles[0].x + origin, self.particles[1].x + origin
        pygame.draw.line(surface, (0, 0, 0), (float(a[0]), float(a[1])), (float(b[0]), float(b[1])))