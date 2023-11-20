from PositionApproximation import constructPositionFunction
from Singleton import Singleton

import jax.numpy as jnp
import jax

from constraints.functions.ConstraintFunctions import ConstraintFunctions


class DistanceConstraintFunctions:
    __metaclass__ = Singleton

    @staticmethod
    @jax.jit
    def constraint(x: jnp.ndarray, params: dict):
        aPosition = x[0]
        bPosition = x[1]
        return jnp.sum((aPosition - bPosition) ** 2) / 2 - (params["distance"] ** 2) / 2

    @staticmethod
    @jax.jit
    def constraintTime(t: jnp.float64, x: jnp.ndarray, params: dict):
        a = x[0]
        b = x[1]
        positionApproximationA = constructPositionFunction(a[0], a[1], a[2])
        positionApproximationB = constructPositionFunction(b[0], b[1], b[2])
        return DistanceConstraintFunctions.constraint(
            jnp.array([positionApproximationA(t), positionApproximationB(t)]),
            params)  # TODO fix this array() call

    def __init__(self):
        self.constraintTimeOptimized, self.dConstraintTime, self.dConstraint, self.d2Constraint =\
            ConstraintFunctions.computeDerivatives(DistanceConstraintFunctions.constraintTime)