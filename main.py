import numpy as np
from typing import List

from constraints.CircleConstraint import CircleConstraint
from constraints.Constraint import Constraint
from constraints.DistanceConstraint import DistanceConstraint
from Particle import Particle
from Simulation import Simulation
from UI import UI
from constraints.functions.CircleConstraintFunctions import CircleConstraintFunctions
from constraints.functions.DistanceConstraintFunctions import DistanceConstraintFunctions


def case1():
    """
    Circle constrant single particle
    """
    particles: List[Particle] = [
        Particle(0, np.array([25, 0], dtype=np.float64))
    ]

    constraints: List[Constraint] = [
        CircleConstraint(0, particles[0], np.array([50, 20], dtype=np.float64), np.float64(100))
    ]

    def force(t: np.float64) -> np.ndarray:
        return np.array([[0, 0]], dtype=np.float64)

    return particles, constraints, force


def case2():
    """
    Distance constraint single particle
    """
    particles: List[Particle] = [
        Particle(0, np.array([50, 25], dtype=np.float64)),
        Particle(1, np.array([50, 50], dtype=np.float64)),
        Particle(2, np.array([25, 0], dtype=np.float64)),
        Particle(3, np.array([0, 0], dtype=np.float64)),
    ]

    constraints: List[Constraint] = [
        DistanceConstraint(0, particles[0], particles[1], np.float64(100)),
        DistanceConstraint(1, particles[2], particles[3], np.float64(100)),
    ]

    def force(t: np.float64) -> np.ndarray:
        return np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float64)

    return particles, constraints, force


def case3():
    """
    Circle and distance constraints multi particles
    """
    particles: List[Particle] = [
        Particle(0, np.array([0, 0], dtype=np.float64)),
        Particle(1, np.array([25, 25], dtype=np.float64))
    ]

    constraints: List[Constraint] = [
        CircleConstraint(0, particles[0], np.array([25, 0], dtype=np.float64), np.float64(100)),
        DistanceConstraint(1, particles[0], particles[1], np.float64(20)),
    ]

    def force(t: np.float64) -> np.ndarray:
        return np.array([[0, 0], [0, 0]], dtype=np.float64)

    return particles, constraints, force


def case4():
    """
    Distance constraints multi particles
    """
    particles: List[Particle] = [
        Particle(0, np.array([0, 0], dtype=np.float64)),
        Particle(1, np.array([25, -25], dtype=np.float64)),
        Particle(2, np.array([50, 0], dtype=np.float64)),
        Particle(3, np.array([75, -25], dtype=np.float64)),
        Particle(4, np.array([100, 0], dtype=np.float64)),
        Particle(5, np.array([125, -25], dtype=np.float64)),
        Particle(6, np.array([150, 0], dtype=np.float64)),
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

    def force(t: np.float64) -> np.ndarray:
        return np.array([[0, 0] for i in range(len(particles))], dtype=np.float64)

    return particles, constraints, force


def case5():
    """
    Distance constraints multi particles
    """
    particles: List[Particle] = [
        Particle(0, np.array([0, 0], dtype=np.float64)),
        Particle(1, np.array([25, -25], dtype=np.float64)),
        Particle(2, np.array([50, 0], dtype=np.float64)),
        Particle(3, np.array([75, -25], dtype=np.float64)),
        Particle(4, np.array([100, 0], dtype=np.float64)),
        Particle(5, np.array([125, -25], dtype=np.float64)),
        Particle(6, np.array([150, 0], dtype=np.float64)),
    ]

    constraints: List[Constraint] = [
        CircleConstraint(0, particles[0], np.array([0, 0], dtype=np.float64), np.float64(0)),
        CircleConstraint(1, particles[6], np.array([150, 0], dtype=np.float64), np.float64(0)),
        DistanceConstraint(2, particles[0], particles[1], np.float64(50)),
        DistanceConstraint(3, particles[0], particles[2], np.float64(50)),
        DistanceConstraint(4, particles[1], particles[2], np.float64(50)),
        DistanceConstraint(5, particles[1], particles[3], np.float64(50)),
        DistanceConstraint(6, particles[2], particles[3], np.float64(50)),
        DistanceConstraint(7, particles[2], particles[4], np.float64(50)),
        DistanceConstraint(8, particles[3], particles[4], np.float64(50)),
        DistanceConstraint(9, particles[3], particles[5], np.float64(50)),
        DistanceConstraint(10, particles[4], particles[5], np.float64(50)),
        DistanceConstraint(11, particles[4], particles[6], np.float64(50)),
        DistanceConstraint(12, particles[5], particles[6], np.float64(50)),
    ]

    def force(t: np.float64) -> np.ndarray:
        return np.array([[0, 0] for i in range(len(particles))], dtype=np.float64)

    return particles, constraints, force


def case6():
    """
    Circle constrant single particle
    """
    particles: List[Particle] = [
        Particle(
            0,
            np.array([25, 0], dtype=np.float64),
        )
    ]

    constraints: List[Constraint] = [
        CircleConstraint(0, particles[0], np.array([50, 20], dtype=np.float64), np.float64(100)),
        CircleConstraint(1, particles[0], np.array([100, 20], dtype=np.float64), np.float64(100))
    ]

    def force(t: np.float64) -> np.ndarray:
        return np.array([[0, 0]], dtype=np.float64)

    return particles, constraints, force


def main():
    print("Loading derivatives...", end="")
    CircleConstraintFunctions()
    DistanceConstraintFunctions()
    print("Done")

    timestep = (np.float64(0.0001))
    particles, constraints, force = case5()
    simulation = Simulation(particles, constraints, timestep, force, False)
    ui = UI(particles, constraints, simulation, timestep)
    ui.run()


if __name__ == '__main__':
    main()
