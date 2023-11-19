import numpy as np

from Particle import Particle
from constraints.Constraint import Constraint
from constraints.functions.CircleConstraintFunctions import CircleConstraintFunctions
from drawers.Drawer import Drawer


class CircleConstraint(Constraint):
    radius: np.float64

    def __init__(self, index: int, particle: Particle, center: np.ndarray, radius: np.float64):
        super().__init__(index, [particle], CircleConstraintFunctions().constraintTimeOptimized,
                         CircleConstraintFunctions().dConstraintTime, CircleConstraintFunctions().dConstraint,
                         CircleConstraintFunctions().d2Constraint)
        self.center, self.radius = center, radius
        self.drawer = None

    def initDrawer(self):
        from drawers.CircleConstraintDrawer import CircleConstraintDrawer
        self.drawer = CircleConstraintDrawer(self)

    def getArgs(self) -> dict:
        return {
            "center": self.center,
            "radius": self.radius
        }

    def getDrawer(self) -> Drawer:
        return self.drawer
