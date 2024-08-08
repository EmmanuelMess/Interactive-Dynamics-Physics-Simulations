from abc import ABC, abstractmethod

import numpy as np


class Drawer(ABC):
    """
    Allows something to be drawn on a surface
    """
    @abstractmethod
    def draw(self, surface: 'pygame.Surface', origin: np.ndarray) -> None:  # noqa: F821
        """
        Override this method to draw on the surface
        """
        pass

    @abstractmethod
    def getText(self) -> str:
        """
        Text to draw behind the simulation
        """
        pass
