from typing import List, TypeVar

from simulator.IndexedElement import IndexedElement


T = TypeVar('T', bound=IndexedElement)


def indexer(list: List[T]) -> List[T]:
    for index, element in enumerate(list):
        element.setIndex(index)

    return list
