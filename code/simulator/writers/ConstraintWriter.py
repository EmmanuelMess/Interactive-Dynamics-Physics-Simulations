import numpy as np
from typing_extensions import List, Union, Dict

from simulator.constraints.Constraint import Constraint
from simulator.writers.Writer import Writer


class ConstraintWriter(Writer):
    def __init__(self, constraint: Constraint) -> None:
        self.constraint = constraint

    def write(self, datapoints: List[Dict[str, Union[int, np.float64]]]) -> None:
        data = {
            'constraint_index': self.constraint.index,
            'c': np.float64(self.constraint.C),
            'dC': np.float64(self.constraint.dC),
        }
        for index in [particle.index for particle in self.constraint.particles]:
            data[f"constraint_{self.constraint.index}_particle_{index}_J_x"] = self.constraint.J[index, 0]
            data[f"constraint_{self.constraint.index}_particle_{index}_J_y"] = self.constraint.J[index, 1]
            data[f"constraint_{self.constraint.index}_particle_{index}_dJ_x"] = self.constraint.dJ[index, 0]
            data[f"constraint_{self.constraint.index}_particle_{index}_dJ_y"] = self.constraint.dJ[index, 1]

        datapoints.append(data)
