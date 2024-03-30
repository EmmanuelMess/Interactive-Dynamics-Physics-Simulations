from simulator.PositionApproximation import constructPositionFunction
from simulator.Singleton import Singleton

import jax.numpy as jnp

from simulator.constraints.functions.ConstraintFunctions import ConstraintFunctions


class CircleConstraintFunctions:
    __metaclass__ = Singleton

    @staticmethod
    def constraint(x: jnp.ndarray, params: dict) -> jnp.float64:
        pPosition = x[0]
        return jnp.sum((pPosition - params["center"]) ** 2) / 2 - (params["radius"] ** 2) / 2

    @staticmethod
    def constraintOfTime(t: jnp.float64, x: jnp.ndarray, v: jnp.ndarray, a: jnp.ndarray, params: dict) -> jnp.float64:
        positionApproximation = constructPositionFunction(x[0], v[0], a[0])
        return CircleConstraintFunctions.constraint(jnp.array([positionApproximation(t)]), params)  # TODO fix this array() call

    def __init__(self):
        self.constraintAndDerivativeOfTime, self.dConstraint, self.d2Constraint = (
            ConstraintFunctions.computeDerivatives(CircleConstraintFunctions.constraintOfTime))
