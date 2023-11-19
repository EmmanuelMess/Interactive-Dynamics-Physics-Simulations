from typing import List, Callable

import numpy as np
from scipy.optimize import root

from SimulationFunctions import SimulationFunctions
from constraints.Constraint import Constraint
from Particle import Particle
from drawers.Drawable import Drawable


class Simulation(Drawable):
    def __init__(self, particles: List[Particle], constraints: List[Constraint], timestep: np.float64,
                 force: Callable[[np.float64], np.ndarray], printData: bool = False):
        super().__init__()
        self.particles, self.constraints, self.timestep, self.force, self.printData = particles, constraints, timestep, force, printData
        self.t = np.float64(0)
        self.error = np.float64(0)

    def initDrawer(self):
        from drawers.SimulationDrawer import SimulationDrawer
        self.setDrawer(SimulationDrawer(self))

    def update(self):
        if self.printData:
            print("----------")
            print("t", self.t)

        if self.printData:
            for particle in self.particles:
                print("i", particle.index, "x", particle.x, "v", particle.v)

        for particle in self.particles:
            particle.aApplied = self.force(self.t)[particle.index].copy()
            particle.a = particle.aApplied.copy()

        dq, Q, W, J, dJ, C, dC, lagrange = SimulationFunctions.matrices(self.particles, self.constraints)

        res = root(lagrange, x0=np.zeros(len(self.constraints), dtype=np.float64), method='lm')

        aConstraint = SimulationFunctions.precompiledForceCalculation(J, res.x)

        self.error = np.sqrt(np.sum(lagrange(res.x)**2))

        if self.printData:
            print("dq", dq)
            print("Q", Q)
            print("W", W)
            print("J", J)
            print("dJ", dJ)

            print("J W J.T", J @ W @ J.T)
            print("dJ dq", dJ @ dq)
            print("JWQ", J @ W @ Q)
            print("C", C)
            print("dC", dC)

            print("l", res.x)
            print("f", lagrange(res.x))

        for particle in self.particles:
            particle.aConstraint = aConstraint[particle.index].copy()
            particle.a = particle.aApplied + particle.aConstraint

            if self.printData:
                print("i", particle.index, "~a + ^a = a", particle.aApplied, particle.aConstraint, particle.a)

            particle.x = SimulationFunctions.x(particle.x, particle.v, particle.a, self.t)
            particle.v = SimulationFunctions.dx(particle.x, particle.v, particle.a, self.t)

        self.t += self.timestep

    def getRunningTime(self):
        return self.t
