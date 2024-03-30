import numpy as np
import pygame

from simulator.Simulation import Simulation
from simulator.drawers.Drawer import Drawer


class SimulationDrawer(Drawer):
    def __init__(self, simulation: Simulation):
        self.simulation = simulation

    def draw(self, surface: pygame.Surface, origin: np.ndarray) -> None:
        pass

    def getText(self) -> str:
        return (f"t {self.simulation.getRunningTime()}s\n"
                f"error {self.simulation.error}\n"
                f"ΔT {self.simulation.updateTiming*1000}ms")
