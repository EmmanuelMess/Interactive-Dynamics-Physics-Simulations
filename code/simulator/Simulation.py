from timeit import default_timer as timer
from typing import Callable, List

import numpy as np

from simulator.SimulationFunctions import SimulationFunctions
from simulator.constraints.Constraint import Constraint
from simulator.Particle import Particle
from simulator.drawers.Drawable import Drawable


class Simulation(Drawable):  # pylint: disable=too-many-instance-attributes
    """
    Manage the state and step running for simulation
    """

    def __init__(self, particles: List[Particle], constraints: List[Constraint],
                 force: Callable[[np.float64], np.ndarray], printData: bool = False):
        super().__init__()
        self.particles = particles
        self.constraints = constraints
        self.force = force
        self.printData = printData
        self.updateTiming: float = 0
        self.ks = np.float64(1000)
        self.kd = np.sqrt(4 * self.ks)
        self.t = np.float64(0)
        self.lastSecondIterations: List[float] = []
        self.error = "0"

    def initDrawer(self) -> None:
        from simulator.drawers.SimulationDrawer import SimulationDrawer

        super(Simulation, self).setDrawer(SimulationDrawer(self))

    def generateGraph(self, grapher: 'Graph') -> None:  # type: ignore[name-defined]  # noqa: F821
        def acceleration(x: np.ndarray, v: np.ndarray) -> np.ndarray:
            particle = self.particles[0]
            particle.x = x
            particle.v = v

            if not particle.static:
                particle.aApplied = self.force(self.t)[particle.index].copy()
                particle.a = particle.aApplied.copy()

            dq, Q, C, dC, W, J, dJ = SimulationFunctions.matrices(self.particles, self.constraints)

            aConstraint, l, f, g = SimulationFunctions.precompiledMinimizeAndForceCalculation(self.ks, self.kd, dq, Q,
                                                                                              C, dC, W, J, dJ)
            self.error = f"constraint {np.linalg.norm(self.ks * C + self.kd * dC)} solve {np.linalg.norm(g * l + f)}"

            return aConstraint[particle.index]

        grapher.draw(acceleration, self.constraints, self.particles)

    def update(self, timestep: np.float64) -> None:  # pylint: disable=too-many-locals
        """
        Run internal simulation update.
        :param timestep: Delta time at which the *next* step will be shown
        """

        if self.printData:
            print("----------")
            print("t", self.t)
            for particle in self.particles:
                print("i", particle.index, "x", particle.x, "v", particle.v)

        start = timer()

        for particle in self.particles:
            if particle.static:
                continue

            particle.aApplied = self.force(self.t)[particle.index].copy()
            particle.a = particle.aApplied.copy()

        dq, Q, C, dC, W, J, dJ = SimulationFunctions.matrices(self.particles, self.constraints)

        aConstraint, l, f, g = SimulationFunctions.precompiledMinimizeAndForceCalculation(self.ks, self.kd, dq, Q, C,
                                                                                          dC, W, J, dJ)
        self.error = f"constraint {np.linalg.norm(self.ks * C + self.kd * dC)} solve {np.linalg.norm(g * l + f)}"

        for particle in self.particles:
            if particle.static:
                continue

            particle.aConstraint = aConstraint[particle.index].copy()
            particle.a = particle.aApplied + particle.aConstraint
            particle.x = SimulationFunctions.x(particle.x, particle.v, particle.a, timestep)
            particle.v = SimulationFunctions.dx(particle.x, particle.v, particle.a, timestep)

        end = timer()

        if self.printData:
            print("ks", self.ks)
            print("kd", self.kd)
            print("J", J)
            print("dJ dq + J W Q + ks C + kd dC", f)
            print("J W J.T", g)
            print("λ", l)
            print("Λ(λ)", g * l.T + f)

            for particle in [particle for particle in self.particles if not particle.static]:
                print("i", particle.index, "~a + ^a = a", particle.aApplied, particle.aConstraint, particle.a)

        self.updateTiming = end - start
        self.lastSecondIterations = [oldEnd for oldEnd in self.lastSecondIterations if end-oldEnd < 1]+[end]
        self.t += timestep

    def getRunningTime(self) -> np.float64:  # pylint: disable=missing-function-docstring
        return self.t
