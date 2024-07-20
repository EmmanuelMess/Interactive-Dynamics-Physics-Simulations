from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import List, Callable

from simulator.Particle import Particle
from simulator.constraints.Constraint import Constraint


class Graph(ABC):
    @abstractmethod
    def draw(self, acceleration: Callable[[np.ndarray, np.ndarray], np.ndarray], constraints: List[Constraint],
             particles: List[Particle]) -> None:
        pass
