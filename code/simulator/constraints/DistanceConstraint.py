import numpy as np

from simulator.Particle import Particle
from simulator.constraints.Constraint import Constraint
from simulator.constraints.functions.DistanceConstraintFunctions import DistanceConstraintFunctions


class DistanceConstraint(Constraint):
    distance: np.float64

    def __init__(self, particleA: Particle, particleB: Particle, distance: np.float64):
        super().__init__([particleA, particleB], DistanceConstraintFunctions().constraintAndDerivativeOfTime,
                         DistanceConstraintFunctions().dConstraint, DistanceConstraintFunctions().d2Constraint)
        self.distance = distance

    def initDrawer(self):
        from simulator.drawers.DistanceConstraintDrawer import DistanceConstraintDrawer
        self.setDrawer(DistanceConstraintDrawer(self))

    def getArgs(self) -> dict:
        return {
            "distance": self.distance
        }
