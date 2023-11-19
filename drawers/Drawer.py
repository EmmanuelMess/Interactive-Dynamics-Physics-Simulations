from abc import ABC, abstractmethod

import numpy as np
import pygame


class Drawer(ABC):
    """
    Allows something to be drawn on a surface
    """
    @abstractmethod
    def draw(self, surface: pygame.Surface, origin: np.ndarray):
        pass

    @abstractmethod
    def getText(self) -> str:
        pass
