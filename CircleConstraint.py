from typing import Any, Callable

import jax.numpy as jnp
from jax import jit, grad

import numpy as np
import pygame

import PositionApproximation
from Constraint import Constraint
from Particle import Particle
from PositionApproximation import constructPositionFunction


class CircleConstraint(Constraint):
    radius: np.float64

    def __init__(self, index: int, particle: Particle, center: np.ndarray, radius: np.float64):
        super().__init__(index, [particle], CircleConstraint.constraintTime)
        self.center, self.radius = center, radius

    @staticmethod
    @jit
    def constraint(center: jnp.ndarray, radius: jnp.ndarray, x: jnp.ndarray) -> jnp.float64:
        return jnp.sum((x - center) ** 2) / 2 - (radius ** 2) / 2

    @staticmethod
    @jit
    def constraintTime(t: jnp.float64, x: jnp.ndarray, params) -> jnp.float64:
        p = x[0]
        positionApproximation = constructPositionFunction(p[0], p[1], p[2])
        return CircleConstraint.constraint(params["center"], params["radius"], positionApproximation(t))

    def getArgs(self) -> dict:
        return {
            "center": self.center,
            "radius": self.radius
        }

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