from abc import ABC, abstractmethod

import numpy as np

from typing_extensions import List, Union, Dict


class Writer(ABC):

    @abstractmethod
    def write(self, datapoints: List[Dict[str, Union[int, np.float64]]]) -> None:
        pass
