from abc import abstractmethod

from simulator.writers.Writer import Writer


class Writable:
    def __init__(self) -> None:
        self.writer: Writer

    @abstractmethod
    def initWriter(self) -> None:
        pass

    def setWriter(self, writer: Writer) -> None:
        self.writer = writer

    def getWriter(self) -> Writer:
        return self.writer
