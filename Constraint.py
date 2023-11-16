from abc import ABC, abstractmethod

from typing import List, Callable

import jax.numpy as jnp
from jax import jit, grad, jacfwd

import numpy as np
import pygame

from Particle import Particle


class Constraint(ABC):
    particles: List[Particle]
    index: int

    @abstractmethod
    def __init__(self, index: int, particles: List[Particle],
                 constraint: Callable[[jnp.ndarray, dict], jnp.float64],
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
        self.dConstraint = jacfwd(self.constraintTime, argnums=1)
        self.d2Constraint = jacfwd(self.dConstraintTime, argnums=1)


    @abstractmethod
    def getArgs(self) -> dict:
        pass

    def getFullParticleMatrix(self) -> jnp.ndarray:
        particleMatrix = np.empty((len(self.particles), 3, 2))

        for i, particle in enumerate(self.particles):
            particleMatrix[i, 0] = particle.x
            particleMatrix[i, 1] = particle.v
            particleMatrix[i, 2] = particle.a

        return jnp.array(particleMatrix)

    def getParticlePositionMatrix(self) -> jnp.ndarray:
        particleMatrix = np.empty((len(self.particles), 2))

        for i, particle in enumerate(self.particles):
            particleMatrix[i] = particle.x

        return jnp.array(particleMatrix)

    def C(self) -> np.float64:
        return self.constraintTime(jnp.float64(0), self.getFullParticleMatrix(), self.getArgs())

    def dC(self) -> np.float64:
        return self.dConstraintTime(jnp.float64(0), self.getFullParticleMatrix(), self.getArgs())

    def d2C(self) -> np.float64:
        return self.d2ConstraintTime(jnp.float64(0), self.getFullParticleMatrix(), self.getArgs())

    def J(self) -> dict:
        constraintJacobian = self.dConstraint(jnp.float64(0), self.getParticlePositionMatrix(), self.getArgs())

        return { particle.i: constraintJacobian[particle.i][0] for particle in self.particles }

    def dJ(self) -> dict:
        constraintJacobian = self.d2Constraint(jnp.float64(0), self.getParticlePositionMatrix(), self.getArgs())

        return {particle.i: constraintJacobian[particle.i][0] for particle in self.particles}

    @abstractmethod
    def surface(self, surface: pygame.Surface, origin: np.ndarray):
        pass
