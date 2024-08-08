from typing import List, TypeVar

from simulator.IndexedElement import IndexedElement


T = TypeVar('T', bound=IndexedElement)


def indexer(l: List[T]) -> List[T]:
    for index, element in enumerate(l):
        element.setIndex(index)

    return l
