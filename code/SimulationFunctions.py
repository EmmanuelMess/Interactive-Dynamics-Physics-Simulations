from typing import List

import numba
import numpy as np

from IndexerIterator import IndexerIterator
from Particle import Particle
from constraints.Constraint import Constraint


class SimulationFunctions:
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
    def matrices(particles: IndexerIterator[Particle], constraints: IndexerIterator[Constraint], weight: np.float64 = 1):
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

        lagrange = lambda l: SimulationFunctions.precompiledLagrange(dq, Q, W, J, dJ, C, dC, ks, kd, l)

        return dq, Q, W, J, dJ, C, dC, lagrange

    @staticmethod
    def x(p, v, a, t):
        return p + v * t + (1/2) * a * t**2

    @staticmethod
    def dx(p, v, a, t):
        return v + a * t
