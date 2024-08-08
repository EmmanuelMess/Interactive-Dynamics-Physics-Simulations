from abc import ABC, abstractmethod

import numpy as np


class Drawer(ABC):
    """
    Allows something to be drawn on a surface
    """
    @abstractmethod
    def draw(self, surface: 'pygame.Surface', origin: np.ndarray) -> None:  # noqa: F821
        pass

    @abstractmethod
    def getText(self) -> str:
        pass
