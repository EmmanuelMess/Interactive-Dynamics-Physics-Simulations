import numpy as np

from Particle import Particle
from constraints.Constraint import Constraint
from constraints.functions.CircleConstraintFunctions import CircleConstraintFunctions
from drawers.Drawer import Drawer


class CircleConstraint(Constraint):
    radius: np.float64

    def __init__(self, particle: Particle, center: np.ndarray, radius: np.float64):
        super().__init__([particle], CircleConstraintFunctions().constraintAndDerivativeOfTime,
                         CircleConstraintFunctions().dConstraint, CircleConstraintFunctions().d2Constraint)
        self.center = center
        self.radius = radius

    def initDrawer(self):
        from drawers.CircleConstraintDrawer import CircleConstraintDrawer
        self.setDrawer(CircleConstraintDrawer(self))

    def getArgs(self) -> dict:
        return {
            "center": self.center,
            "radius": self.radius
        }

