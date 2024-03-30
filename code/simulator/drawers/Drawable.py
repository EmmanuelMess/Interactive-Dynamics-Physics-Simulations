from abc import abstractmethod

from simulator.drawers.Drawer import Drawer


class Drawable:
    def __init__(self):
        self.drawer = None

    @abstractmethod
    def initDrawer(self):
        pass

    def setDrawer(self, drawer: Drawer):
        self.drawer = drawer

    def getDrawer(self) -> Drawer:
        return self.drawer
