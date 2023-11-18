from typing import List, Callable

import numpy as np
from scipy.optimize import root
import numba

from Constraint import Constraint
from Particle import Particle


class Simulation:
    def __init__(self, particles: List[Particle], constraints: List[Constraint], timestep: np.float64,
                 force: Callable[[np.float64], np.ndarray], printData: bool = False):
        self.particles, self.constraints, self.timestep, self.force, self.printData = particles, constraints, timestep, force, printData
        self.t = np.float64(0)
        self.error = np.float64(0)

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

        dq, Q, W, J, dJ, C, dC, lagrange = Simulation.matrices(self.particles, self.constraints)

        res = root(lagrange, x0=np.zeros(len(self.constraints), dtype=np.float64), method='lm')

        aConstraint = Simulation.precompiledForceCalculation(J, res.x)

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

            particle.x = Simulation.x(particle.x, particle.v, particle.a, self.t)
            particle.v = Simulation.dx(particle.x, particle.v, particle.a, self.t)

        self.t += self.timestep

    def getRunningTime(self):
        return self.t

    @staticmethod
    @numba.jit(nopython=True)
    def precompiledForceCalculation(J: np.ndarray, l: np.float64) -> np.ndarray:
        return (J.T @ l).reshape((-1, 2))

    @staticmethod
    @numba.jit(nopython=True)
    def precompiledLagrange(dq: np.ndarray, Q: np.ndarray, W: np.ndarray, J: np.ndarray, dJ: np.ndarray, C: np.ndarray,
                            dC: np.ndarray, ks: np.float64, kd: np.float64, l: np.float64):
        return ((J @ W @ J.T) * l.T + dJ @ dq + J @ W @ Q + ks * C + kd * dC).reshape((-1,))

    @staticmethod
    def matrices(particles: List[Particle], constraints: List[Constraint], weight: np.float64 = 1):
        d = 2
        n = len(particles)
        m = len(constraints)
        ks = np.float64(0.1)
        kd = np.float64(1)

        dq = np.zeros((n, d), dtype=np.float64)
        Q = np.zeros((n, d), dtype=np.float64)
        C = np.zeros((m,), dtype=np.float64)
        dC = np.zeros((m,), dtype=np.float64)
        W = np.identity(n * d, dtype=np.float64) * weight
        J = np.zeros((m, n, d), dtype=np.float64)
        dJ = np.zeros((m, n, d), dtype=np.float64)

        for particle in particles:
            dq[particle.index] = particle.v
            Q[particle.index] = particle.a

        dq = dq.reshape((n * d, 1))
        Q = Q.reshape((n * d, 1))

        for constraint in constraints:
            C[constraint.index] += constraint.C()
            dC[constraint.index] += constraint.dC()
            JForConstraint = constraint.J()
            dJForConstraint = constraint.dJ()
            for j, particle in enumerate(constraint.particles):
                J[constraint.index, particle.index] += JForConstraint[j]
                dJ[constraint.index, particle.index] += dJForConstraint[j]

        J = J.reshape((m, n * d))
        dJ = dJ.reshape((m, n * d))

        lagrange = lambda l: Simulation.precompiledLagrange(dq, Q, W, J, dJ, C, dC, ks, kd, l)

        return dq, Q, W, J, dJ, C, dC, lagrange

    @staticmethod
    def x(p, v, a, t):
        return p + v * t + (1/2) * a * t**2

    @staticmethod
    def dx(p, v, a, t):
        return v + a * t
