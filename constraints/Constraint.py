from abc import ABC, abstractmethod

from typing import List, Callable

import jax.numpy as jnp

import numpy as np

from Particle import Particle
from drawers.Drawable import Drawable


class Constraint(ABC, Drawable):
    particles: List[Particle]
    index: int

    @abstractmethod
    def __init__(self, index: int, particles: List[Particle],
                 constraintTime: Callable[[jnp.float64, jnp.ndarray, dict], jnp.float64],
                 dConstraintTime: Callable[[jnp.float64, jnp.ndarray, dict], jnp.float64],
                 dConstraint: Callable[[jnp.float64, jnp.ndarray, dict], jnp.ndarray],
                 d2Constraint: Callable[[jnp.float64, jnp.ndarray, dict], jnp.ndarray]):
        """
        :param index: Index of this particle
        :param particles: All particles that are affected by this constraint
        For the rest of the parameters pass the result from Constraint.computeDerivatives
        """
        super().__init__()
        self.index, self.particles = index, particles
        self.constraintTime, self.dConstraintTime, self.dConstraint, self.d2Constraint = constraintTime, dConstraintTime, dConstraint, d2Constraint

    @abstractmethod
    def getArgs(self) -> dict:
        pass

    def getFullParticleMatrix(self) -> jnp.ndarray:
        particleMatrix = np.empty((len(self.particles), 3, 2), dtype=np.float64)

        for i, particle in enumerate(self.particles):
            particleMatrix[i, 0] = particle.x
            particleMatrix[i, 1] = particle.v
            particleMatrix[i, 2] = particle.a

        return jnp.array(particleMatrix)

    def C(self) -> np.float64:
        return self.constraintTime(jnp.float64(0), self.getFullParticleMatrix(), self.getArgs())

    def dC(self) -> np.float64:
        return self.dConstraintTime(jnp.float64(0), self.getFullParticleMatrix(), self.getArgs())

    def J(self) -> np.array:
        constraintJacobian = self.dConstraint(jnp.float64(0), self.getFullParticleMatrix(), self.getArgs())
        return constraintJacobian[:, 0]  # Only get the position derivative  # TODO jit this select

    def dJ(self) -> np.array:
        constraintJacobian = self.d2Constraint(jnp.float64(0), self.getFullParticleMatrix(), self.getArgs())
        return constraintJacobian[:, 0]  # Only get the position derivative  # TODO jit this select

