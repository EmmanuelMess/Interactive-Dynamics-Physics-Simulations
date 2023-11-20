from PositionApproximation import constructPositionFunction
from Singleton import Singleton

import jax.numpy as jnp
import jax

from constraints.functions.ConstraintFunctions import ConstraintFunctions


class CircleConstraintFunctions:
    __metaclass__ = Singleton

    @staticmethod
    @jax.jit
    def constraint(x: jnp.ndarray, params: dict) -> jnp.float64:
        pPosition = x[0]
        return jnp.sum((pPosition - params["center"]) ** 2) / 2 - (params["radius"] ** 2) / 2

    @staticmethod
    @jax.jit
    def constraintTime(t: jnp.float64, x: jnp.ndarray, params: dict) -> jnp.float64:
        p = x[0]
        positionApproximation = constructPositionFunction(p[0], p[1], p[2])
        return CircleConstraintFunctions.constraint(jnp.array([positionApproximation(t)]), params)  # TODO fix this array() call

    def __init__(self):
        self.constraintTimeOptimized, self.dConstraintTime, self.dConstraint, self.d2Constraint = (
            ConstraintFunctions.computeDerivatives(CircleConstraintFunctions.constraintTime))
