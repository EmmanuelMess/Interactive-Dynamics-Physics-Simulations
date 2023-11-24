from itertools import count

import numpy as np
from typing import List

import pygame

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
        Particle(0, np.array([0, 0], dtype=np.float64), static=True),
        Particle(1, np.array([25, -25], dtype=np.float64)),
        Particle(2, np.array([50, 0], dtype=np.float64)),
        Particle(3, np.array([75, -25], dtype=np.float64)),
        Particle(4, np.array([100, 0], dtype=np.float64)),
        Particle(5, np.array([125, -25], dtype=np.float64)),
        Particle(6, np.array([150, 0], dtype=np.float64), static=True),
    ]

    constraints: List[Constraint] = [
        DistanceConstraint(0, particles[0], particles[1], np.float64(50)),
        DistanceConstraint(1, particles[0], particles[2], np.float64(50)),
        DistanceConstraint(2, particles[1], particles[2], np.float64(50)),
        DistanceConstraint(3, particles[1], particles[3], np.float64(50)),
        DistanceConstraint(4, particles[2], particles[3], np.float64(50)),
        DistanceConstraint(5, particles[2], particles[4], np.float64(50)),
        DistanceConstraint(6, particles[3], particles[4], np.float64(50)),
        DistanceConstraint(7, particles[3], particles[5], np.float64(50)),
        DistanceConstraint(8, particles[4], particles[5], np.float64(50)),
        DistanceConstraint(9, particles[4], particles[6], np.float64(50)),
        DistanceConstraint(10, particles[5], particles[6], np.float64(50)),
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


def case7():
    """
    Distance constraints in a grid for a lot of particles
    """
    CONSTRAINT_DISTANCE = np.float64(100)
    DISTANCE = 50
    EXTERNAL_GRID_WIDTH = 4
    INTERNAL_GRID_WIDTH = EXTERNAL_GRID_WIDTH-1

    externalGrid = range(0, DISTANCE*EXTERNAL_GRID_WIDTH, DISTANCE)
    internalGrid = range(DISTANCE//2, DISTANCE*INTERNAL_GRID_WIDTH, DISTANCE)
    positionsGridA = [np.array([x, y], dtype=np.float64) for x in externalGrid for y in externalGrid]
    positionsGridB = [np.array([x, y], dtype=np.float64) for x in internalGrid for y in internalGrid]

    particles: List[Particle] = []

    for i, xy in enumerate(list(positionsGridA)+list(positionsGridB)):
        particles.append(Particle(i, np.array(xy, dtype=np.float64)))

    constraints: List[Constraint] = []

    M = len(positionsGridA)
    K = EXTERNAL_GRID_WIDTH
    N = INTERNAL_GRID_WIDTH

    index = 0
    for i in range(len(positionsGridB)):
        constraints.append(DistanceConstraint(index, particles[i+i//N], particles[i+M], CONSTRAINT_DISTANCE))
        constraints.append(DistanceConstraint(index+1, particles[i+i//N+1], particles[i+M], CONSTRAINT_DISTANCE))
        constraints.append(DistanceConstraint(index+2, particles[i+i//N+K], particles[i+M], CONSTRAINT_DISTANCE))
        constraints.append(DistanceConstraint(index+3, particles[i+i//N+K+1], particles[i+M], CONSTRAINT_DISTANCE))
        index += 4

    def force(t: np.float64) -> np.ndarray:
        return np.array([[0, 0] for i in range(len(particles))], dtype=np.float64)

    return particles, constraints, force


def case8():
    """
    Distance constraints in a grid for a lot of particles
    """
    DISTANCE = 50
    CONSTRAINT_DISTANCE = np.float64(np.sqrt(DISTANCE**2+DISTANCE**2)/2)
    EXTERNAL_GRID_WIDTH = 4
    INTERNAL_GRID_WIDTH = EXTERNAL_GRID_WIDTH-1

    externalGrid = range(0, DISTANCE*EXTERNAL_GRID_WIDTH, DISTANCE)
    internalGrid = range(DISTANCE//2, DISTANCE*INTERNAL_GRID_WIDTH, DISTANCE)
    positionsGridA = [np.array([x, y], dtype=np.float64) for x in externalGrid for y in externalGrid]
    positionsGridB = [np.array([x, y], dtype=np.float64) for x in internalGrid for y in internalGrid]

    particles: List[Particle] = []

    for i, xy in enumerate(list(positionsGridA)+list(positionsGridB)):
        particles.append(Particle(i, np.array(xy, dtype=np.float64)))

    constraints: List[Constraint] = []

    M = len(positionsGridA)
    K = EXTERNAL_GRID_WIDTH
    N = INTERNAL_GRID_WIDTH

    index = 0
    for i in range(len(positionsGridB)):
        constraints.append(DistanceConstraint(index, particles[i+i//N], particles[i+M], CONSTRAINT_DISTANCE))
        constraints.append(DistanceConstraint(index+1, particles[i+i//N+1], particles[i+M], CONSTRAINT_DISTANCE))
        constraints.append(DistanceConstraint(index+2, particles[i+i//N+K], particles[i+M], CONSTRAINT_DISTANCE))
        constraints.append(DistanceConstraint(index+3, particles[i+i//N+K+1], particles[i+M], CONSTRAINT_DISTANCE))

        constraints.append(DistanceConstraint(index+4, particles[i+i//N], particles[i+i//N+1], DISTANCE))
        constraints.append(DistanceConstraint(index+5, particles[i+i//N+1], particles[i+i//N+K+1], DISTANCE))
        constraints.append(DistanceConstraint(index+6, particles[i+i//N+K+1], particles[i+i//N+K], DISTANCE))
        constraints.append(DistanceConstraint(index+7, particles[i+i//N+K], particles[i+i//N], DISTANCE))

        index += 8

    def force(t: np.float64) -> np.ndarray:
        return np.array([[10*np.abs(np.sin(1000*t)), -10*np.abs(np.sin(1000*t))]] + [[0, 0] for i in range(len(particles)-1)], dtype=np.float64)

    return particles, constraints, force


def run(simulation: Simulation, ui: UI):
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        simulation.update()

        ui.update()

    pygame.quit()


def main():
    print("Loading derivatives...", end="")
    CircleConstraintFunctions()
    DistanceConstraintFunctions()
    print("Done")

    timestep = (np.float64(0.0001))
    particles, constraints, force = case8()
    simulation = Simulation(particles, constraints, timestep, force, False)
    ui = UI([simulation]+particles+constraints, timestep)
    run(simulation, ui)


if __name__ == '__main__':
    main()