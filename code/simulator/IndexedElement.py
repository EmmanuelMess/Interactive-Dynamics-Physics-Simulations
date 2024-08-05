from dataclasses import dataclass


class IndexedElement:
    index: int

    def setIndex(self, index: int) -> None:
        self.index = index
