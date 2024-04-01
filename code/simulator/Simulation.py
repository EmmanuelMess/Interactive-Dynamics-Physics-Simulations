from timeit import default_timer as timer
from typing import Callable

import numpy as np
from scipy.optimize import root  # type: ignore

from simulator.IndexerIterator import IndexerIterator
from simulator.SimulationFunctions import SimulationFunctions
from simulator.constraints.Constraint import Constraint
from simulator.Particle import Particle
from simulator.drawers.Drawable import Drawable


class Simulation(Drawable):  # pylint: disable=too-many-instance-attributes
    """
    Manage the state and step running for simulation
    """

    def __init__(self, particles: IndexerIterator[Particle], constraints: IndexerIterator[Constraint],
                 timestep: np.float64, force: Callable[[np.float64], np.ndarray], printData: bool = False):
        super().__init__()
        self.particles = particles
        self.constraints = constraints
        self.timestep = timestep
        self.force = force
        self.printData = printData
        self.updateTiming: float = 0
        self.t = np.float64(0)
        self.error = np.float64(0)

    def initDrawer(self) -> None:
        from simulator.drawers.SimulationDrawer import SimulationDrawer

        super(Simulation, self).setDrawer(SimulationDrawer(self))

    def update(self) -> None:
        """
        Run internal simulation update.
        :param timestep: Delta time at which the *next* step will be shown
        """

        start = timer()

        if self.printData:
            print("----------")
            print("t", self.t)

        if self.printData:
            for particle in self.particles:
                print("i", particle.index, "x", particle.x, "v", particle.v)

        for particle in self.particles:
            if particle.static:
                continue

            particle.aApplied = self.force(self.t)[particle.index].copy()
            particle.a = particle.aApplied.copy()

        f, g, J = SimulationFunctions.matrices(self.particles, self.constraints)

        # Solve for λ in g λ = -f, where f = dJ dq + J W Q + ks C + kd dC and g = J W J.T
        l, residuals, _, _ = np.linalg.lstsq(g, -f)

        aConstraint = SimulationFunctions.precompiledForceCalculation(J, l)

        self.error = np.sum(residuals)

        if self.printData:
            print("J", J)
            print("dJ dq + J W Q + ks C + kd dC", f)
            print("J W J.T", g)
            print("λ", l)
            print("Λ(λ)", g * l.T + f)

        for particle in self.particles:
            if particle.static:
                continue

            particle.aConstraint = aConstraint[particle.index].copy()
            particle.a = particle.aApplied + particle.aConstraint

            if self.printData:
                print("i", particle.index, "~a + ^a = a", particle.aApplied, particle.aConstraint, particle.a)

            particle.x = SimulationFunctions.x(particle.x, particle.v, particle.a, self.t)
            particle.v = SimulationFunctions.dx(particle.x, particle.v, particle.a, self.t)

        end = timer()

        self.updateTiming = end - start
        self.t += self.timestep

    def getRunningTime(self) -> np.float64:  # pylint: disable=missing-function-docstring
        return self.t
