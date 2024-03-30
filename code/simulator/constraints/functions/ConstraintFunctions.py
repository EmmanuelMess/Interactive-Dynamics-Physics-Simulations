from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import jacfwd, grad, value_and_grad


class ConstraintFunctions:
    @staticmethod
    def computeDerivatives(
            constraintTime: Callable[[jnp.float64, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict], jnp.float64]) \
            -> Tuple[
                Callable[[jnp.float64, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict], jnp.float64],
                Callable[[jnp.float64, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict], jnp.float64],
                Callable[[jnp.float64, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict], jnp.float64]
            ]:
        """
        :param constraintTime: A function of time that can be derived on t=0 to obtain the constraints, the second
        parameter is getParticleMatrix() and the third is getArgs(). The function passed should be pure and precompiled
        """
        constraintAndDerivativeOfTime = jax.jit(value_and_grad(constraintTime, argnums=0))
        dConstraint = jax.jit(jacfwd(constraintTime, argnums=1))
        d2Constraint = jax.jit(jacfwd(grad(constraintTime, argnums=0), argnums=1))
        return constraintAndDerivativeOfTime, dConstraint, d2Constraint
