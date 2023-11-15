from abc import ABC, abstractmethod

from typing import List, Callable

import jax.numpy as jnp
from jax import jit, grad

import numpy as np
import pygame

from Particle import Particle


class Constraint(ABC):
    particles: List[Particle]
    index: int

    @abstractmethod
    def __init__(self, index: int, particles: List[Particle],
                 constraintTime: Callable[[jnp.float64, jnp.ndarray, dict], jnp.float64]):
        """

        :param index: Index of this particle
        :param particles: All particles that are affected by this constraint
        :param constraintTime: A function of time that can be derived on t=0 to obtain the constraints, the second
        parameter is getParticleMatrix() and the third is getArgs(). The function passed should be pure and precompiled
        """
        self.index, self.particles, self.constraintTime = index, particles, constraintTime
        self.dConstraintTime = grad(self.constraintTime, argnums=0)
        self.d2ConstraintTime = grad(self.dConstraintTime, argnums=0)


    @abstractmethod
    def getArgs(self) -> dict:
        pass

    def getParticleMatrix(self) -> jnp.ndarray:
        particleMatrix = np.empty((len(self.particles), 3, 2))

        for i, particle in enumerate(self.particles):
            particleMatrix[i, 0] = particle.x
            particleMatrix[i, 1] = particle.v
            particleMatrix[i, 2] = particle.a

        return jnp.array(particleMatrix)

    def C(self, particleMatrix: jnp.ndarray) -> np.float64:
        return self.constraintTime(jnp.float64(0), particleMatrix, self.getArgs())

    def dC(self, particleMatrix: jnp.ndarray) -> np.float64:
        return self.dConstraintTime(jnp.float64(0), particleMatrix, self.getArgs())

    def d2C(self, particleMatrix: jnp.ndarray) -> np.float64:
        return self.d2ConstraintTime(jnp.float64(0), particleMatrix, self.getArgs())

    @abstractmethod
    def J(self) -> dict:
        pass

    @abstractmethod
    def dJ(self) -> dict:
        pass

    @abstractmethod
    def surface(self, surface: pygame.Surface, origin: np.ndarray):
        pass
