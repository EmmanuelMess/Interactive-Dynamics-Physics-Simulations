from typing import List, Callable

import numpy as np
from scipy.optimize import root

from Constraint import Constraint
from Particle import Particle


class Simulation:
    def __init__(self, particles: List[Particle], constraints: List[Constraint], timestep: np.float64,
                 force: Callable[[np.float64], np.ndarray]):
        self.particles, self.constraints, self.timestep, self.force = particles, constraints, timestep, force
        self.t = np.float64(0)

    def update(self):
        print("----------")
        print("t", self.t)

        for particle in self.particles:
            print("i", particle.i, "x", particle.x, "v", particle.v)

        for particle in self.particles:
            particle.aApplied = self.force(self.t)[particle.i]
            particle.a = particle.aApplied.copy()

        dq, Q, W, J, dJ, C, dC, lagrange = Simulation.matrices(self.particles, self.constraints)

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

        res = root(lagrange, x0=np.zeros(len(self.constraints)), method='lm')

        print("l", res.x)
        print("f", lagrange(res.x))

        aConstraint = (J.T @ res.x).reshape((-1, 2))

        for particle in self.particles:
            particle.aConstraint = aConstraint[particle.i].copy()
            particle.a = particle.aApplied + particle.aConstraint

            print("i", particle.i, "~a + ^a = a", particle.aApplied, particle.aConstraint, particle.a)

            particle.x = Simulation.x(particle.x, particle.v, particle.a, self.t)
            particle.v = Simulation.dx(particle.x, particle.v, particle.a, self.t)

        self.t += self.timestep

    def getRunningTime(self):
        return self.t

    @staticmethod
    def matrices(particles: List[Particle], constraints: List[Constraint]):
        d = 2
        n = len(particles)
        m = len(constraints)
        ks = 0.0001
        kd = 0.0001

        dq = np.zeros((n, d))
        Q = np.zeros((n, d))
        C = np.zeros((m, 1))
        dC = np.zeros((m, 1))
        W = np.identity(n * d)
        J = np.zeros((m, n, d))
        dJ = np.zeros((m, n, d))

        for particle in particles:
            for k in range(d):
                dq[particle.i] = particle.v.copy()
                Q[particle.i] = particle.a.copy()

        dq = dq.reshape((n * d, 1))
        Q = Q.reshape((n * d, 1))

        for constraint in constraints:
            particlesMatrix = constraint.getParticleMatrix()
            C[constraint.index, 0] = constraint.C(particlesMatrix)
            dC[constraint.index, 0] = constraint.dC(particlesMatrix)
            for j, value in constraint.J().items():
                J[constraint.index, j] = value.copy()
            for j, value in constraint.dJ().items():
                dJ[constraint.index, j] = value.copy()

        J = J.reshape((m, n * d))
        dJ = dJ.reshape((m, n * d))

        def lagrange(l):
            return ((J @ W @ J.T) * l.T + dJ @ dq + J @ W @ Q + ks * C + kd * dC).reshape((-1,))

        return dq, Q, W, J, dJ, C, dC, lagrange

    @staticmethod
    def x(p, v, a, t):
        return p + v * t + (1/2) * a * t**2

    @staticmethod
    def dx(p, v, a, t):
        return v + a * t
