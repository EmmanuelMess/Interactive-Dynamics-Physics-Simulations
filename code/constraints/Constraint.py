from abc import ABC, abstractmethod

from typing import List, Callable, Tuple

import jax.numpy as jnp
import jax

import numpy as np

import PositionApproximation
from IndexedElement import IndexedElement
from Particle import Particle
from drawers.Drawable import Drawable


class Constraint(ABC, Drawable, IndexedElement):
    particles: List[Particle]

    @abstractmethod
    def __init__(self, particles: List[Particle],
                 constraintAndDerivativeOfTime: Callable[[jnp.float64, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict], Tuple[jnp.float64, jnp.float64]],
                 dConstraint: Callable[[jnp.float64, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict], jnp.ndarray],
                 d2Constraint: Callable[[jnp.float64, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict], jnp.ndarray]):
        """
        :param index: Index of this particle
        :param particles: All particles that are affected by this constraint
        For the rest of the parameters pass the result from Constraint.computeDerivatives
        """
        super().__init__()
        self.particles = particles
        self.constraintAndDerivativeOfTime = constraintAndDerivativeOfTime
        self.dConstraint = dConstraint
        self.d2Constraint = d2Constraint

    @abstractmethod
    def getArgs(self) -> dict:
        pass

    def getFullParticleMatrices(self):
        positionMatrix = jnp.array([particle.x for particle in self.particles])
        velocityMatrix = jnp.array([particle.v for particle in self.particles])
        accelerationMatrix = jnp.array([particle.a for particle in self.particles])

        return positionMatrix, velocityMatrix, accelerationMatrix

    def get(self) -> Tuple[jnp.float64, jnp.float64, jnp.ndarray, jnp.ndarray]:
        x, v, a = self.getFullParticleMatrices()
        args = self.getArgs()
        C, dC = self.constraintAndDerivativeOfTime(PositionApproximation.ZERO_TIME, x, v, a, args)
        J = self.dConstraint(PositionApproximation.ZERO_TIME, x, v, a, args)
        dJ = self.d2Constraint(PositionApproximation.ZERO_TIME, x, v, a, args)
        return C, dC, J, dJ
