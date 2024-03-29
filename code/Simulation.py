from timeit import default_timer as timer
from typing import List, Callable

import numpy as np
from scipy.optimize import root

from IndexerIterator import IndexerIterator
from SimulationFunctions import SimulationFunctions
from constraints.Constraint import Constraint
from Particle import Particle
from drawers.Drawable import Drawable


class Simulation(Drawable):
    def __init__(self, particles: IndexerIterator[Particle], constraints: IndexerIterator[Constraint], timestep: np.float64,
                 force: Callable[[np.float64], np.ndarray], printData: bool = False):
        super().__init__()
        self.particles = particles
        self.constraints = constraints
        self.timestep = timestep
        self.force = force
        self.printData = printData
        self.updateTiming = 0
        self.t = np.float64(0)
        self.error = np.float64(0)

    def initDrawer(self):
        from drawers.SimulationDrawer import SimulationDrawer
        self.setDrawer(SimulationDrawer(self))

    def update(self):
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

        lagrangeArgs, lagrange, J = SimulationFunctions.matrices(self.particles, self.constraints)
        f, g = lagrangeArgs

        res = root(lagrange, x0=np.zeros(len(self.constraints), dtype=np.float64), method='lm', args=lagrangeArgs)

        aConstraint = SimulationFunctions.precompiledForceCalculation(J, res.x)

        self.error = np.sqrt(np.sum(lagrange(res.x, *lagrangeArgs)**2))

        if self.printData:
            print("J", J)
            print("dJ dq + J W Q + ks C + kd dC", f)
            print("J W J.T", g)

            print("l", res.x)
            print("f", lagrange(res.x))

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

    def getRunningTime(self):
        return self.t
