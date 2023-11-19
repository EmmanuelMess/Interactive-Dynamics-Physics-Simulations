import numpy as np

from Particle import Particle
from constraints.Constraint import Constraint
from constraints.functions.DistanceConstraintFunctions import DistanceConstraintFunctions
from drawers.Drawer import Drawer


class DistanceConstraint(Constraint):
    distance: np.float64

    def __init__(self, index: int, particleA: Particle, particleB: Particle, distance: np.float64):
        super().__init__(index, [particleA, particleB], DistanceConstraintFunctions().constraintTimeOptimized,
                         DistanceConstraintFunctions().dConstraintTime, DistanceConstraintFunctions().dConstraint,
                         DistanceConstraintFunctions().d2Constraint)
        self.distance = distance
        self.drawer = None

    def initDrawer(self):
        from drawers.DistanceConstraintDrawer import DistanceConstraintDrawer
        self.drawer = DistanceConstraintDrawer(self)

    def getArgs(self) -> dict:
        return {
            "distance": self.distance
        }

    def getDrawer(self) -> Drawer:
        return self.drawer