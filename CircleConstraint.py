import jax.numpy as jnp
from jax import jacfwd, grad

import numpy as np
import pygame

import PositionApproximation
from Constraint import Constraint
from Particle import Particle
from PositionApproximation import constructPositionFunction


class CircleConstraint(Constraint):
    radius: np.float64

    def __init__(self,  i: int, particle: Particle, center: np.ndarray, radius: np.float64):
        super().__init__(i, [particle])
        self.center, self.radius = center, radius

    @staticmethod
    def constraint(radius: jnp.float64, center: jnp.ndarray):
        def f(x):
            return jnp.sum((x - center) ** 2) / 2 - (radius ** 2) / 2

        return f

    def C(self) -> np.float64:
        p = self.particles[0]
        positionApproximation = constructPositionFunction(jnp.array(p.x), jnp.array(p.v), jnp.array(p.a))
        constraintFunction = lambda t: CircleConstraint.constraint(self.radius, jnp.array(self.center))(positionApproximation(t))
        c = constraintFunction(jnp.float64(0))
        return np.float64(c)

    def dC(self) -> np.float64:
        p = self.particles[0]
        positionApproximation = constructPositionFunction(jnp.array(p.x), jnp.array(p.v), jnp.array(p.a))
        constraintFunction = lambda t: CircleConstraint.constraint(self.radius, jnp.array(self.center))(positionApproximation(t))
        c = grad(constraintFunction)(jnp.float64(0))
        return np.float64(c)

    def d2C(self) -> np.float64:
        p = self.particles[0]
        positionApproximation = constructPositionFunction(jnp.array(p.x), jnp.array(p.v), jnp.array(p.a))
        constraintFunction = lambda t: CircleConstraint.constraint(self.radius, jnp.array(self.center))(positionApproximation(t))
        c = grad(grad(constraintFunction))(jnp.float64(0))
        return np.float64(c)

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