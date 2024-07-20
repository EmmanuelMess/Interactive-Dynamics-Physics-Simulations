import numpy as np
from matplotlib import pyplot as plt
from typing_extensions import Callable, List

from simulator import Constants
from simulator.Particle import Particle
from simulator.constraints.Constraint import Constraint
from simulator.graphs.Graph import Graph


class AccelerationPortrait(Graph):
    def __init__(self, velocity: np.ndarray, divisions: int = 100):
        self.velocity = velocity
        self.divisions = divisions

    def draw(self, acceleration: Callable[[np.ndarray, np.ndarray], np.ndarray], constraints: List[Constraint],
             particles: List[Particle]) -> None:
        print("Drawing plot...", end="")
        X, Y = np.meshgrid(np.linspace(-Constants.WIDTH // 2, Constants.WIDTH // 2, self.divisions),
                           np.linspace(-Constants.HEIGHT // 2, Constants.HEIGHT // 2, self.divisions))
        u, v = np.zeros_like(X), np.zeros_like(X)
        NI, NJ = X.shape

        for i in range(NI):
            for j in range(NJ):
                x = X[i, j]
                y = Y[i, j]
                fp = acceleration(np.array([x, y], dtype=np.float64), self.velocity)
                u[i, j] = fp[0]
                v[i, j] = fp[1]

        plt.title(f"Constant velocity {self.velocity}")
        plt.streamplot(X, Y, u, v)
        plt.axis('square')
        plt.axis((-float(Constants.WIDTH) / 2.0, float(Constants.WIDTH) / 2.0, -float(Constants.HEIGHT) / 2.0,
                  float(Constants.HEIGHT) / 2.0))
        plt.show()
        print("Done")
