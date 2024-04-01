import argparse
import typing

import numpy as np

import pygame
from typing_extensions import List

from simulator import Cases
from simulator.Particle import Particle
from simulator.Simulation import Simulation
from simulator.UI import UI
from simulator.constraints.functions.CircleConstraintFunctions import CircleConstraintFunctions
from simulator.constraints.functions.DistanceConstraintFunctions import DistanceConstraintFunctions
import importlib.util

from simulator.drawers.Drawable import Drawable

if importlib.util.find_spec("scalene") is not None:
    from scalene import scalene_profiler  # type: ignore

def run(simulation: Simulation, ui: UI) -> None:
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        simulation.update()

        ui.update()

    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--case', required=True)
    parser.add_argument('-p', '--profile', action='store_true')
    args = parser.parse_args()

    print("Loading derivatives...", end="")
    CircleConstraintFunctions()
    DistanceConstraintFunctions()
    print("Done")

    timestep = (np.float64(0.00001))
    particles, constraints, force = Cases.CASES[args.case]()
    simulation = Simulation(particles, constraints, timestep, force, False)
    drawables = [typing.cast(Drawable, simulation)]\
            +typing.cast(List[Drawable], particles)\
            +typing.cast(List[Drawable], constraints)
    ui = UI(drawables, timestep)

    # HACK First update will compile everything and is not representative for profiling
    simulation.update()

    if args.profile and importlib.util.find_spec("scalene") is not None:
        scalene_profiler.start()

    run(simulation, ui)


if __name__ == '__main__':
    main()
