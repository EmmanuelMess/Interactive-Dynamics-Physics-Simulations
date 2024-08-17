import argparse
import importlib.util
import sys
import typing
from timeit import default_timer as timer

import numpy as np
import pygame
from typing_extensions import List, Union

from simulator import Cases, Constants
from simulator.FileSaver import FileSaver
from simulator.Simulation import Simulation
from simulator.UI import UI
from simulator.constraints.functions.CircleConstraintFunctions import CircleConstraintFunctions
from simulator.constraints.functions.DistanceConstraintFunctions import DistanceConstraintFunctions
from simulator.drawers.Drawable import Drawable

from simulator.writers.Writable import Writable

if importlib.util.find_spec("scalene") is not None:
    from scalene import scalene_profiler  # type: ignore


def runWithUi(simulation: Simulation, ui: UI, fileSaver: Union[FileSaver, None], timestep: np.float64) -> None:
    running = True

    lastFrame = timer()
    while running:
        userHasntQuit = ui is None or len([event for event in pygame.event.get() if event.type == pygame.QUIT]) == 0
        running = userHasntQuit

        while timer() - lastFrame < 1.0/Constants.FPS:
            simulation.update(timestep)

        ui.update()
        if fileSaver is not None:
            fileSaver.update()

        lastFrame = timer()


def runNoUi(simulation: Simulation, fileSaver: FileSaver, timestep: np.float64, iterations: int) -> None:
    for i in range(0, iterations):
        simulation.update(timestep)

        print(f"\r{i+1}/{iterations}...", end="")

        if fileSaver is not None:
            fileSaver.update()



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True, choices=Cases.CASES.keys(), help='Which scenario to execute')
    parser.add_argument('--profile', action='store_true', help='Print which functions take the most time')
    parser.add_argument('--no_ui', action='store_true', default=False, help='Run without ui')
    parser.add_argument('--save', metavar="FILENAME", type=str, default=None, help='Save to file')
    parser.add_argument('--iterations', type=int, default=None, help='How many iterations to run for')
    args = parser.parse_args()

    if args.no_ui and not args.save:
        print("Error: Not save to file and no UI", file=sys.stderr)
        return

    if not args.no_ui and args.iterations is not None:
        print("Error: Iterations doesn't work with UI", file=sys.stderr)
        return

    print("Loading derivatives...", end="")
    CircleConstraintFunctions()
    DistanceConstraintFunctions()
    print("Done")

    timestep = np.float64(0.001)
    particles, constraints, force = Cases.CASES[args.case]()
    simulation = Simulation(particles, constraints, force, False)
    # simulation.generateGraph(AccelerationPortrait(np.array([0, 0], dtype=np.float64)))
    # simulation.generateGraph(CostGraph(np.array([0, 0], dtype=np.float64)))

    if args.save is None:
        saver = None
    else:
        writables = typing.cast(List[Writable], particles)
        saver = FileSaver(f"output/{args.save}", writables)

    if args.no_ui:
        ui = None
    else:
        drawables = [typing.cast(Drawable, simulation)]
        drawables += typing.cast(List[Drawable], particles)
        drawables += typing.cast(List[Drawable], constraints)

        ui = UI(drawables)

    if args.profile:
        if importlib.util.find_spec("scalene") is None:
            print("Error: scalene not installed!")

        # HACK First update will compile everything and is not representative for profiling
        simulation.update(timestep)
        scalene_profiler.start()

    if args.no_ui:
        runNoUi(simulation, saver, timestep, args.iterations)
    else:
        runWithUi(simulation, ui, saver, timestep)

    if not args.no_ui:
        pygame.quit()

    if args.save is not None:
        saver.save()


if __name__ == '__main__':
    main()
