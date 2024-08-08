from abc import ABC, abstractmethod

from typing import List, Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from simulator import PositionApproximation
from simulator.IndexedElement import IndexedElement
from simulator.Particle import Particle
from simulator.drawers.Drawable import Drawable


class Constraint(ABC, Drawable, IndexedElement):
    @abstractmethod
    def __init__(self, particles: List[Particle],
                 constraintAndDerivativeOfTime: Callable[[jnp.float64, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict],
                 Tuple[jnp.float64, jnp.float64]],
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

    @staticmethod
    @jax.jit
    def getFullParticleMatrices(particlesX: List[np.ndarray], particlesV: List[np.ndarray],
                                particlesA: List[np.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        positionMatrix = jnp.array(particlesX, dtype=np.float64)
        velocityMatrix = jnp.array(particlesV, dtype=np.float64)
        accelerationMatrix = jnp.array(particlesA, dtype=np.float64)

        return positionMatrix, velocityMatrix, accelerationMatrix

    def get(self) -> Tuple[jnp.float64, jnp.float64, jnp.ndarray, jnp.ndarray]:
        x, v, a = Constraint.getFullParticleMatrices([particle.x for particle in self.particles],
                                                     [particle.v for particle in self.particles],
                                                     [particle.a for particle in self.particles])
        args = self.getArgs()
        C, dC = self.constraintAndDerivativeOfTime(PositionApproximation.ZERO_TIME, x, v, a, args)
        J = self.dConstraint(PositionApproximation.ZERO_TIME, x, v, a, args)
        dJ = self.d2Constraint(PositionApproximation.ZERO_TIME, x, v, a, args)
        return C, dC, J, dJ
