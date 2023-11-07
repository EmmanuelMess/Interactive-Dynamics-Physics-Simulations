from abc import ABC, abstractmethod

from typing import List

import numpy as np
import pygame

from Particle import Particle


class Constraint(ABC):
    particles: List[Particle]
    i: int

    @abstractmethod
    def __init__(self, i: int, particles: List[Particle]):
        self.i, self.particles = i, particles

    @abstractmethod
    def C(self) -> np.float64:
        pass

    @abstractmethod
    def dC(self) -> np.float64:
        pass

    @abstractmethod
    def J(self) -> dict:
        pass

    @abstractmethod
    def dJ(self) -> dict:
        pass

    @abstractmethod
    def surface(self, surface: pygame.Surface, origin: np.ndarray):
        pass
