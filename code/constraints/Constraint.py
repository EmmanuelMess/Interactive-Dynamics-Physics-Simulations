from abc import ABC, abstractmethod

from typing import List, Callable, Tuple

import jax.numpy as jnp

import numpy as np

from IndexedElement import IndexedElement
from Particle import Particle
from drawers.Drawable import Drawable


class Constraint(ABC, Drawable, IndexedElement):
    particles: List[Particle]

    @abstractmethod
    def __init__(self, particles: List[Particle],
                 constraintTime: Callable[[jnp.float64, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict], jnp.float64],
                 dConstraintTime: Callable[[jnp.float64, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict], jnp.float64],
                 dConstraint: Callable[[jnp.float64, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict], jnp.ndarray],
                 d2Constraint: Callable[[jnp.float64, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict], jnp.ndarray]):
        """
        :param index: Index of this particle
        :param particles: All particles that are affected by this constraint
        For the rest of the parameters pass the result from Constraint.computeDerivatives
        """
        super().__init__()
        self.particles = particles
        self.constraintTime = constraintTime
        self.dConstraintTime = dConstraintTime
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

    def C(self) -> np.float64:
        x, v, a = self.getFullParticleMatrices()
        return self.constraintTime(jnp.float64(0), x, v, a, self.getArgs())

    def dC(self) -> np.float64:
        x, v, a = self.getFullParticleMatrices()
        return self.dConstraintTime(jnp.float64(0), x, v, a, self.getArgs())

    def J(self) -> np.array:
        x, v, a = self.getFullParticleMatrices()
        return self.dConstraint(jnp.float64(0), x, v, a, self.getArgs())

    def dJ(self) -> np.array:
        x, v, a = self.getFullParticleMatrices()
        return self.d2Constraint(jnp.float64(0), x, v, a, self.getArgs())

