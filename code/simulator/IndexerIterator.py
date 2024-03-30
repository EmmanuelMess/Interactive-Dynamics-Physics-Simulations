from typing import List, TypeVar

from simulator.IndexedElement import IndexedElement


T = TypeVar('T', bound=IndexedElement)


class IndexerIterator(List[T]):
    def __init__(self, list: List[T]):
        super().__init__(list)

        for index, element in enumerate(self):
            element.index = index
