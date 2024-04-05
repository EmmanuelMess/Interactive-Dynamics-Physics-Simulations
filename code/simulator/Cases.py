from typing import List

import numpy as np
from typing_extensions import Callable, Tuple, Dict

from simulator.IndexerIterator import IndexerIterator
from simulator.Particle import Particle
from simulator.constraints.CircleConstraint import CircleConstraint
from simulator.constraints.Constraint import Constraint
from simulator.constraints.DistanceConstraint import DistanceConstraint


def case1() -> Tuple[IndexerIterator[Particle], IndexerIterator[Constraint], Callable[[np.float64], np.ndarray]]:
    """
    Circle constrant single particle
    """
    particles: IndexerIterator[Particle] = IndexerIterator([
        Particle(np.array([25, 0], dtype=np.float64))
    ])

    constraints: IndexerIterator[Constraint] = IndexerIterator([
        CircleConstraint(particles[0], np.array([50, 20], dtype=np.float64), np.float64(100))
    ])

    def force(t: np.float64) -> np.ndarray:
        return np.array([[0, 0]], dtype=np.float64)

    return particles, constraints, force


def case2() -> Tuple[IndexerIterator[Particle], IndexerIterator[Constraint], Callable[[np.float64], np.ndarray]]:
    """
    Distance constraint single particle
    """
    particles: IndexerIterator[Particle] = IndexerIterator([
        Particle(np.array([50, 25], dtype=np.float64)),
        Particle(np.array([50, 50], dtype=np.float64)),
        Particle(np.array([25, 0], dtype=np.float64)),
        Particle(np.array([0, 0], dtype=np.float64)),
    ])

    constraints: IndexerIterator[Constraint] = IndexerIterator([
        DistanceConstraint(particles[0], particles[1], np.float64(100)),
        DistanceConstraint(particles[2], particles[3], np.float64(100)),
    ])

    def force(t: np.float64) -> np.ndarray:
        return np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float64)

    return particles, constraints, force


def case3() -> Tuple[IndexerIterator[Particle], IndexerIterator[Constraint], Callable[[np.float64], np.ndarray]]:
    """
    Circle and distance constraints multi particles
    """
    particles: IndexerIterator[Particle] = IndexerIterator([
        Particle(np.array([0, 0], dtype=np.float64)),
        Particle(np.array([25, 25], dtype=np.float64))
    ])

    constraints: IndexerIterator[Constraint] = IndexerIterator([
        CircleConstraint(particles[0], np.array([25, 0], dtype=np.float64), np.float64(100)),
        DistanceConstraint(particles[0], particles[1], np.float64(20)),
    ])

    def force(t: np.float64) -> np.ndarray:
        return np.array([[0, 0], [0, 0]], dtype=np.float64)

    return particles, constraints, force


def case4() -> Tuple[IndexerIterator[Particle], IndexerIterator[Constraint], Callable[[np.float64], np.ndarray]]:
    """
    Distance constraints multi particles
    """
    particles: IndexerIterator[Particle] = IndexerIterator([
        Particle(np.array([0, 0], dtype=np.float64)),
        Particle(np.array([25, -25], dtype=np.float64)),
        Particle(np.array([50, 0], dtype=np.float64)),
        Particle(np.array([75, -25], dtype=np.float64)),
        Particle(np.array([100, 0], dtype=np.float64)),
        Particle(np.array([125, -25], dtype=np.float64)),
        Particle(np.array([150, 0], dtype=np.float64)),
    ])

    constraints: IndexerIterator[Constraint] = IndexerIterator([
        DistanceConstraint(particles[0], particles[1], np.float64(25)),
        DistanceConstraint(particles[0], particles[2], np.float64(25)),
        DistanceConstraint(particles[1], particles[2], np.float64(25)),
        DistanceConstraint(particles[1], particles[3], np.float64(25)),
        DistanceConstraint(particles[2], particles[3], np.float64(25)),
        DistanceConstraint(particles[2], particles[4], np.float64(25)),
        DistanceConstraint(particles[3], particles[4], np.float64(25)),
        DistanceConstraint(particles[3], particles[5], np.float64(25)),
        DistanceConstraint(particles[4], particles[5], np.float64(25)),
        DistanceConstraint(particles[4], particles[6], np.float64(25)),
        DistanceConstraint(particles[5], particles[6], np.float64(25)),
    ])

    def force(t: np.float64) -> np.ndarray:
        return np.array([[0, 0] for i in range(len(particles))], dtype=np.float64)

    return particles, constraints, force


def case5() -> Tuple[IndexerIterator[Particle], IndexerIterator[Constraint], Callable[[np.float64], np.ndarray]]:
    """
    Distance constraints multi particles
    """
    particles: IndexerIterator[Particle] = IndexerIterator([
        Particle(np.array([0, 0], dtype=np.float64), static=True),
        Particle(np.array([25, -25], dtype=np.float64)),
        Particle(np.array([50, 0], dtype=np.float64)),
        Particle(np.array([75, -25], dtype=np.float64)),
        Particle(np.array([100, 0], dtype=np.float64)),
        Particle(np.array([125, -25], dtype=np.float64)),
        Particle(np.array([150, 0], dtype=np.float64), static=True),
    ])

    constraints: IndexerIterator[Constraint] = IndexerIterator([
        DistanceConstraint(particles[0], particles[1], np.float64(50)),
        DistanceConstraint(particles[0], particles[2], np.float64(50)),
        DistanceConstraint(particles[1], particles[2], np.float64(50)),
        DistanceConstraint(particles[1], particles[3], np.float64(50)),
        DistanceConstraint(particles[2], particles[3], np.float64(50)),
        DistanceConstraint(particles[2], particles[4], np.float64(50)),
        DistanceConstraint(particles[3], particles[4], np.float64(50)),
        DistanceConstraint(particles[3], particles[5], np.float64(50)),
        DistanceConstraint(particles[4], particles[5], np.float64(50)),
        DistanceConstraint(particles[4], particles[6], np.float64(50)),
        DistanceConstraint(particles[5], particles[6], np.float64(50)),
    ])

    def force(t: np.float64) -> np.ndarray:
        return np.array([[0, 0] for i in range(len(particles))], dtype=np.float64)

    return particles, constraints, force


def case6() -> Tuple[IndexerIterator[Particle], IndexerIterator[Constraint], Callable[[np.float64], np.ndarray]]:
    """
    Circle constrant single particle
    """
    particles: IndexerIterator[Particle] = IndexerIterator([Particle(np.array([25, 0], dtype=np.float64),)])

    constraints: IndexerIterator[Constraint] = IndexerIterator([
        CircleConstraint(particles[0], np.array([50, 20], dtype=np.float64), np.float64(100)),
        CircleConstraint(particles[0], np.array([100, 20], dtype=np.float64), np.float64(100))
    ])

    def force(t: np.float64) -> np.ndarray:
        return np.array([[0, 0]], dtype=np.float64)

    return particles, constraints, force


def case7() -> Tuple[IndexerIterator[Particle], IndexerIterator[Constraint], Callable[[np.float64], np.ndarray]]:
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

    for xy in list(positionsGridA)+list(positionsGridB):
        particles.append(Particle(np.array(xy, dtype=np.float64)))

    constraints: List[Constraint] = []

    M = len(positionsGridA)
    K = EXTERNAL_GRID_WIDTH
    N = INTERNAL_GRID_WIDTH

    for i in range(len(positionsGridB)):
        constraints.append(DistanceConstraint(particles[i+i//N], particles[i+M], CONSTRAINT_DISTANCE))
        constraints.append(DistanceConstraint(particles[i+i//N+1], particles[i+M], CONSTRAINT_DISTANCE))
        constraints.append(DistanceConstraint(particles[i+i//N+K], particles[i+M], CONSTRAINT_DISTANCE))
        constraints.append(DistanceConstraint(particles[i+i//N+K+1], particles[i+M], CONSTRAINT_DISTANCE))

    def force(t: np.float64) -> np.ndarray:
        return np.array([[0, 0] for i in range(len(particles))], dtype=np.float64)

    return IndexerIterator(particles), IndexerIterator(constraints), force


def case8() -> Tuple[IndexerIterator[Particle], IndexerIterator[Constraint], Callable[[np.float64], np.ndarray]]:
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

    for xy in list(positionsGridA)+list(positionsGridB):
        particles.append(Particle(np.array(xy, dtype=np.float64)))

    constraints: List[Constraint] = []

    M = len(positionsGridA)
    K = EXTERNAL_GRID_WIDTH
    N = INTERNAL_GRID_WIDTH

    for i in range(len(positionsGridB)):
        constraints.append(DistanceConstraint(particles[i+i//N], particles[i+M], CONSTRAINT_DISTANCE))
        constraints.append(DistanceConstraint(particles[i+i//N+1], particles[i+M], CONSTRAINT_DISTANCE))
        constraints.append(DistanceConstraint(particles[i+i//N+K], particles[i+M], CONSTRAINT_DISTANCE))
        constraints.append(DistanceConstraint(particles[i+i//N+K+1], particles[i+M], CONSTRAINT_DISTANCE))

        constraints.append(DistanceConstraint(particles[i+i//N], particles[i+i//N+1], np.float64(DISTANCE)))
        constraints.append(DistanceConstraint(particles[i+i//N+1], particles[i+i//N+K+1], np.float64(DISTANCE)))
        constraints.append(DistanceConstraint(particles[i+i//N+K+1], particles[i+i//N+K], np.float64(DISTANCE)))
        constraints.append(DistanceConstraint(particles[i+i//N+K], particles[i+i//N], np.float64(DISTANCE)))

    def force(t: np.float64) -> np.ndarray:
        return np.array([[10 * np.abs(np.sin(1000 * t)), -10 * np.abs(np.sin(1000 * t))]]
                        + [[0, 0] for i in range(len(particles) - 1)], dtype=np.float64)

    return IndexerIterator(particles), IndexerIterator(constraints), force

def case9() -> Tuple[IndexerIterator[Particle], IndexerIterator[Constraint], Callable[[np.float64], np.ndarray]]:
    """
    Circle constrants single particle
    """
    particles: IndexerIterator[Particle] = IndexerIterator([Particle(np.array([25, 0], dtype=np.float64),)])

    constraints: IndexerIterator[Constraint] = IndexerIterator([
        CircleConstraint(particles[0], np.array([-125, 20], dtype=np.float64), np.float64(100)),
        CircleConstraint(particles[0], np.array([125, 20], dtype=np.float64), np.float64(100))
    ])

    def force(t: np.float64) -> np.ndarray:
        return np.array([[0, 0]], dtype=np.float64)

    return particles, constraints, force


def case10() -> Tuple[IndexerIterator[Particle], IndexerIterator[Constraint], Callable[[np.float64], np.ndarray]]:
    """
    Distance constraints in a grid for a lot of particles
    """
    CONSTRAINT_DISTANCE = np.float64(150)
    DISTANCE = 50
    EXTERNAL_GRID_WIDTH = 4
    INTERNAL_GRID_WIDTH = EXTERNAL_GRID_WIDTH-1

    externalGrid = range(0, DISTANCE*EXTERNAL_GRID_WIDTH, DISTANCE)
    internalGrid = range(DISTANCE//2, DISTANCE*INTERNAL_GRID_WIDTH, DISTANCE)
    positionsGridA = [np.array([x, y], dtype=np.float64) for x in externalGrid for y in externalGrid]
    positionsGridB = [np.array([x, y], dtype=np.float64) for x in internalGrid for y in internalGrid]

    particles: List[Particle] = []

    for xy in list(positionsGridA)+list(positionsGridB):
        particles.append(Particle(np.array(xy, dtype=np.float64)))

    constraints: List[Constraint] = []

    M = len(positionsGridA)
    K = EXTERNAL_GRID_WIDTH
    N = INTERNAL_GRID_WIDTH

    for i in range(len(positionsGridB)):
        constraints.append(DistanceConstraint(particles[i+i//N], particles[i+M], CONSTRAINT_DISTANCE))
        constraints.append(DistanceConstraint(particles[i+i//N+1], particles[i+M], CONSTRAINT_DISTANCE))
        constraints.append(DistanceConstraint(particles[i+i//N+K], particles[i+M], CONSTRAINT_DISTANCE))
        constraints.append(DistanceConstraint(particles[i+i//N+K+1], particles[i+M], CONSTRAINT_DISTANCE))

    def force(t: np.float64) -> np.ndarray:
        return np.array([[0, 0] for i in range(len(particles))], dtype=np.float64)

    return IndexerIterator(particles), IndexerIterator(constraints), force


CASES: Dict[
    str,
    Callable[[], Tuple[IndexerIterator[Particle], IndexerIterator[Constraint], Callable[[np.float64], np.ndarray]]]
] =\
    {"1": case1, "2": case2, "3": case3, "4": case4, "5": case5, "6": case6, "7": case7, "8": case8, "9": case9,
     "10": case10}
