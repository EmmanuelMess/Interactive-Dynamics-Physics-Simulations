import numpy as np

from Particle import Particle
from constraints.Constraint import Constraint
from constraints.functions.DistanceConstraintFunctions import DistanceConstraintFunctions
from drawers.Drawer import Drawer


class DistanceConstraint(Constraint):
    distance: np.float64

    def __init__(self, particleA: Particle, particleB: Particle, distance: np.float64):
        super().__init__([particleA, particleB], DistanceConstraintFunctions().constraintAndDerivativeOfTime,
                         DistanceConstraintFunctions().dConstraint, DistanceConstraintFunctions().d2Constraint)
        self.distance = distance

    def initDrawer(self):
        from drawers.DistanceConstraintDrawer import DistanceConstraintDrawer
        self.setDrawer(DistanceConstraintDrawer(self))

    def getArgs(self) -> dict:
        return {
            "distance": self.distance
        }
