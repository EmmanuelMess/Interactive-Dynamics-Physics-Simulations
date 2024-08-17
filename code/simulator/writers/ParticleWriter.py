import numpy as np
from typing_extensions import List, Union, Dict

from simulator.Particle import Particle
from simulator.writers.Writer import Writer


class ParticleWriter(Writer):
    def __init__(self, particle: Particle) -> None:
        self.particle = particle

    def write(self, datapoints: List[Dict[str, Union[int, np.float64]]]) -> None:
        data = {
            'index': self.particle.index,
            'position_x': self.particle.x[0],
            'position_y': self.particle.x[1],
            'velocity_x': self.particle.v[0],
            'velocity_y': self.particle.v[1],
            'acceleration_x': self.particle.a[0],
            'acceleration_y': self.particle.a[1],
            'acceleration_constraint_x': self.particle.aConstraint[0],
            'acceleration_constraint_y': self.particle.aConstraint[1],
            'acceleration_applied_x': self.particle.aApplied[0],
            'acceleration_applied_y': self.particle.aApplied[1],
        }
        datapoints.append(data)
