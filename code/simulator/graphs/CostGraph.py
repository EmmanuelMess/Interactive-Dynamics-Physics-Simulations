import numpy as np
from matplotlib import pyplot as plt
from typing_extensions import Callable, List

from simulator import Constants
from simulator.Particle import Particle
from simulator.constraints.Constraint import Constraint
from simulator.graphs.Graph import Graph


class CostGraph(Graph):

    def __init__(self, velocity: np.ndarray, divisions: int = 100):
        self.velocity = velocity
        self.divisions = divisions

    def draw(self, acceleration: Callable[[np.ndarray, np.ndarray], np.ndarray], constraints: List[Constraint],
             particles: List[Particle]) -> None:
        if len(particles) > 1 or len(constraints) > 1:
            return

        particle = particles[0]
        constraint = constraints[0]

        x, y = np.meshgrid(np.linspace(-Constants.WIDTH // 2, Constants.WIDTH // 2, self.divisions),
                           np.linspace(-Constants.HEIGHT // 2, Constants.HEIGHT // 2, self.divisions))

        z = np.zeros(x.shape, dtype=np.float64)
        NI, NJ = x.shape

        for i in range(NI):
            for j in range(NJ):
                particle.x = np.array([x[i, j],  y[i, j]], dtype=np.float64)
                particle.v = self.velocity

                C, _, _, _ = constraint.get()

                z[i, j] = np.maximum(np.minimum(C, 1000), -1000)

        cont = plt.contourf(x, y, z, 100)
        plt.colorbar(cont)
        plt.axis('square')
        plt.axis((-float(Constants.WIDTH) / 2.0, float(Constants.WIDTH) / 2.0, -float(Constants.HEIGHT) / 2.0,
                  float(Constants.HEIGHT) / 2.0))
        plt.show()
