import numpy as np
from typing import List

from Constraint import Constraint
from DistanceConstraint import DistanceConstraint
from Particle import Particle
from CircleConstraint import CircleConstraint
from Simulation import Simulation
from UI import UI


def case1():
    """
    Circle constrant single particle
    """
    particles: List[Particle] = [
        Particle(
            0,
            np.array([25, 0], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64)
        )
    ]

    constraints: List[Constraint] = [
        CircleConstraint(0, particles[0], np.array([50, 20]), np.float64(100))
    ]

    timestep = np.float64(0.001)

    def force(t: np.float64) -> np.ndarray:
        return np.array([[2, 2]])

    return particles, constraints, timestep, force


def case2():
    """
    Distance constraint single particle
    """
    particles: List[Particle] = [
        Particle(
            0,
             np.array([25, 0], dtype=np.float64),
             np.array([0.0, 0.0], dtype=np.float64)
        ),
        Particle(
            1,
            np.array([25, 25], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64)
        )
    ]

    constraints: List[Constraint] = [
        DistanceConstraint(0, particles[0], particles[1], np.float64(100))
    ]

    timestep = np.float64(0.001)

    def force(t: np.float64) -> np.ndarray:
        return np.array([[0, 0], [0, 0]])

    return particles, constraints, timestep, force


def case3():
    """
    Circle and distance constraints multi particles
    """
    particles: List[Particle] = [
        Particle(
            0,
             np.array([25, 0], dtype=np.float64),
             np.array([0.0, 0.0], dtype=np.float64)
        ),
        Particle(
            1,
            np.array([25, 25], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64)
        )
    ]

    constraints: List[Constraint] = [
        CircleConstraint(0, particles[0], np.array([0, 0]), np.float64(100)),
        DistanceConstraint(1, particles[0], particles[1], np.float64(20)),
    ]

    timestep = (np.float64(0.001))

    def force(t: np.float64) -> np.ndarray:
        return np.array([[0, 2], [2, 0]])

    return particles, constraints, timestep, force


def case4():
    """
    Distance constraints multi particles
    """
    particles: List[Particle] = [
        Particle(0, np.array([0, 0], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)),
        Particle(1, np.array([25, -25], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)),
        Particle(2, np.array([50, 0], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)),
        Particle(3, np.array([75, -25], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)),
        Particle(4, np.array([100, 0], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)),
        Particle(5, np.array([125, -25], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)),
        Particle(6, np.array([150, 0], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)),
    ]

    constraints: List[Constraint] = [
        DistanceConstraint(0, particles[0], particles[1], np.float64(25)),
        DistanceConstraint(1, particles[0], particles[2], np.float64(25)),
        DistanceConstraint(2, particles[1], particles[2], np.float64(25)),
        DistanceConstraint(3, particles[1], particles[3], np.float64(25)),
        DistanceConstraint(4, particles[2], particles[3], np.float64(25)),
        DistanceConstraint(5, particles[2], particles[4], np.float64(25)),
        DistanceConstraint(6, particles[3], particles[4], np.float64(25)),
        DistanceConstraint(7, particles[3], particles[5], np.float64(25)),
        DistanceConstraint(8, particles[4], particles[5], np.float64(25)),
        DistanceConstraint(9, particles[4], particles[6], np.float64(25)),
        DistanceConstraint(10, particles[5], particles[6], np.float64(25)),
    ]

    timestep = (np.float64(0.001))

    def force(t: np.float64) -> np.ndarray:
        return np.array([[0, 0] for i in range(len(particles))])

    return particles, constraints, timestep, force


def main():
    particles, constraints, timestep, force = case2()
    simulation = Simulation(particles, constraints, timestep, force)
    ui = UI(particles, constraints, simulation, timestep)
    ui.run()


if __name__ == '__main__':
    main()
