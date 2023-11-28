from abc import ABC, abstractmethod

from typing import List, Callable, Tuple

import jax.numpy as jnp

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

    def getFullParticleMatrices(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        positionMatrix = np.empty((len(self.particles), 2), dtype=np.float64)
        velocityMatrix = np.empty((len(self.particles), 2), dtype=np.float64)
        accelerationMatrix = np.empty((len(self.particles), 2), dtype=np.float64)

        for i, particle in enumerate(self.particles):
            positionMatrix[i] = particle.x
            velocityMatrix[i] = particle.v
            accelerationMatrix[i] = particle.a

        return jnp.array(positionMatrix), jnp.array(velocityMatrix), jnp.array(accelerationMatrix)

    def get(self) -> Tuple[jnp.float64, jnp.float64, jnp.ndarray, jnp.ndarray]:
        x, v, a = self.getFullParticleMatrices()
        args = self.getArgs()
        C, dC = self.constraintAndDerivativeOfTime(PositionApproximation.ZERO_TIME, x, v, a, args)
        J = self.dConstraint(PositionApproximation.ZERO_TIME, x, v, a, args)
        dJ = self.d2Constraint(PositionApproximation.ZERO_TIME, x, v, a, args)
        return C, dC, J, dJ
