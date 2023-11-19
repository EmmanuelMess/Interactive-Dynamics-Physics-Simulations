from typing import Callable

import jax
import jax.numpy as jnp
from jax import jacfwd, grad

from Singleton import Singleton


class ConstraintFunctions:
    @staticmethod
    def computeDerivatives(constraintTime: Callable[[jnp.float64, jnp.ndarray, dict], jnp.float64]):
        """
        :param constraintTime: A function of time that can be derived on t=0 to obtain the constraints, the second
        parameter is getParticleMatrix() and the third is getArgs(). The function passed should be pure and precompiled
        """
        constraintTime = jax.jit(constraintTime)
        dConstraintTime = jax.jit(grad(constraintTime, argnums=0))
        dConstraint = jax.jit(jacfwd(constraintTime, argnums=1))
        d2Constraint = jax.jit(jacfwd(dConstraintTime, argnums=1))
        return constraintTime, dConstraintTime, dConstraint, d2Constraint
