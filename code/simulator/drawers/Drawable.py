from abc import abstractmethod

from simulator.drawers.Drawer import Drawer


class Drawable:
    def __init__(self) -> None:
        self.drawer: Drawer

    @abstractmethod
    def initDrawer(self) -> None:
        pass

    def setDrawer(self, drawer: Drawer) -> None:
        self.drawer = drawer

    def getDrawer(self) -> Drawer:
        return self.drawer
