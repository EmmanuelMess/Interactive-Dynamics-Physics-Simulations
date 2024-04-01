import numba  # type: ignore
import numpy as np

from typing_extensions import Tuple, Callable

from simulator.IndexerIterator import IndexerIterator
from simulator.Particle import Particle
from simulator.constraints.Constraint import Constraint


class SimulationFunctions:
    @staticmethod
    @numba.njit
    def precompiledForceCalculation(J: np.ndarray, l: np.ndarray) -> np.ndarray:
        """
        Resulting force for the particles (see mathematical model)
        """
        return (J.T @ l).reshape((-1, 2))

    @staticmethod
    @numba.njit
    def precompiledMatricesComputation(ks: np.float64, kd: np.float64, dq: np.ndarray, Q: np.ndarray, C: np.ndarray,
                                       dC: np.ndarray, W: np.ndarray, J: np.ndarray, dJ: np.ndarray)\
            -> Tuple[np.ndarray, np.ndarray]:
        f = dJ @ dq + J @ W @ Q + ks * C + kd * dC
        g = J @ W @ J.T
        return f, g

    @staticmethod
    def matrices(particles: IndexerIterator[Particle], constraints: IndexerIterator[Constraint],
                 weight: np.float64 = np.float64(1))\
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # pylint: disable=too-many-locals
        """
        Compute the matrices to run the lagrangian multipliers (see mathematical model)
        """
        d = 2
        n = len(particles)
        m = len(constraints)
        ks = np.float64(0.00001)
        kd = np.float64(0.01)

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

        dq = dq.reshape((n * d,))
        Q = Q.reshape((n * d,))

        for constraint in constraints:
            CForConstraint, dCForConstraint, JForConstraint, dJForConstraint = constraint.get()
            C[constraint.index] += CForConstraint
            dC[constraint.index] += dCForConstraint

            # HACK advanced numpy indexing is much faster than the equivalent loop, as jax copies the array values
            # before returning them when using normal indexing
            indicesConstraint = [constraint.index for _ in constraint.particles]
            indicesParticle = [particle.index for particle in constraint.particles]

            J[indicesConstraint, indicesParticle] += JForConstraint
            dJ[indicesConstraint, indicesParticle] += dJForConstraint

        J = J.reshape((m, n * d))
        dJ = dJ.reshape((m, n * d))

        f, g = SimulationFunctions.precompiledMatricesComputation(ks, kd, dq, Q, C, dC, W, J, dJ)

        return f, g, J

    @staticmethod
    def x(p: np.ndarray, v: np.ndarray, a: np.ndarray, t: np.float64) -> np.ndarray:
        """
        Position Taylor approximation from position, velocity, acceleration and time
        """
        return p + v * t + (1/2) * a * t**2

    @staticmethod
    def dx(p: np.ndarray, v: np.ndarray, a: np.ndarray, t: np.float64) -> np.ndarray:  # pylint: disable=unused-argument
        """
        Derivative of Taylor approximation from position, velocity, acceleration and time
        """
        return v + a * t
