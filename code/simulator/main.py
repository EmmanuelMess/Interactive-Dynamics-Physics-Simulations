import argparse

import numpy as np

import pygame

from simulator import Cases
from simulator.Simulation import Simulation
from simulator.UI import UI
from simulator.constraints.functions.CircleConstraintFunctions import CircleConstraintFunctions
from simulator.constraints.functions.DistanceConstraintFunctions import DistanceConstraintFunctions


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
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--case', required=True)
    parser.add_argument('-p', '--profile', action='store_true')
    args = parser.parse_args()

    print("Loading derivatives...", end="")
    CircleConstraintFunctions()
    DistanceConstraintFunctions()
    print("Done")

    timestep = (np.float64(0.0001))
    particles, constraints, force = Cases.CASES[args.case]()
    simulation = Simulation(particles, constraints, timestep, force, False)
    ui = UI([simulation]+particles+constraints, timestep)

    # HACK First update will compile everything and is not representative for profiling
    simulation.update()

    if args.profile:
        from scalene import scalene_profiler

        scalene_profiler.start()

    run(simulation, ui)


if __name__ == '__main__':
    main()
