import numpy as np
import pandas as pd
from typing_extensions import List, Union, Dict

from simulator.writers.Writable import Writable


class FileSaver:
    def __init__(self, fileName: str, writables: List[Writable]) -> None:
        self.fileName = fileName
        self.writables = writables
        self.datapoints: List[Dict[str, Union[int, np.float64]]] = []

        for writable in self.writables:
            writable.initWriter()

    def update(self) -> None:
        for writable in self.writables:
            writable.getWriter().write(self.datapoints)

    def save(self) -> None:
        pd.DataFrame(self.datapoints).to_csv(f"{self.fileName}.csv")
