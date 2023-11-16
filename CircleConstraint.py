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
        super().__init__(index, [particle], CircleConstraint.constraint, CircleConstraint.constraintTime)
        self.center, self.radius = center, radius

    @staticmethod
    @jit
    def constraint(x: jnp.ndarray, params: dict) -> jnp.float64:
        pPosition = x[0]
        return jnp.sum((pPosition - params["center"]) ** 2) / 2 - (params["radius"] ** 2) / 2

    @staticmethod
    @jit
    def constraintTime(t: jnp.float64, x: jnp.ndarray, params: dict) -> jnp.float64:
        p = x[0]
        positionApproximation = constructPositionFunction(p[0], p[1], p[2])
        return CircleConstraint.constraint(jnp.array([positionApproximation(t)]), params)

    def getArgs(self) -> dict:
        return {
            "center": self.center,
            "radius": self.radius
        }

    def surface(self, surface: pygame.Surface, origin: np.ndarray):
        c = (float((self.center+origin)[0]), float((self.center+origin)[1]))
        r = float(self.radius)
        rect = pygame.Rect(c[0]-r, c[1]-r, r*2, r*2)
        pygame.draw.ellipse(surface, (0, 0, 0), rect, width=1)
