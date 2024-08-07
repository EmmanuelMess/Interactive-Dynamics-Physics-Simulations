from itertools import count

import pygame
import numpy as np
from typing import List

from simulator import Constants
from simulator.Constants import HEIGHT, WIDTH
from simulator.drawers.Drawable import Drawable


class UI:
    def __init__(self, drawables: List[Drawable], timestep: np.float64) -> None:
        self.drawables, self.timestep = drawables, timestep
        self.size = [HEIGHT, WIDTH]
        self.origin = np.array(self.size)/2
        self.running = True

        np.set_printoptions(suppress=True)

        pygame.init()
        self.screen = pygame.display.set_mode(self.size)
        self.font = pygame.font.SysFont("monospace", 11)

        for drawable in self.drawables:
            drawable.initDrawer()

    def showDrawables(self) -> None:
        allText = [f"UI target FPS {Constants.FPS}"]

        for text in self.drawables:
            allText += text.getDrawer().getText().split("\n")

        for yPositionTextLine, string in zip(count(0, 10), allText):
            label = self.font.render(string, 1, (0, 0, 0))
            self.screen.blit(label, (0, yPositionTextLine))

    def update(self) -> None:
        self.screen.fill((255, 255, 255))
        for drawable in self.drawables:
            drawable.getDrawer().draw(self.screen, self.origin)
        pygame.draw.line(self.screen, (0, 0, 0), (self.origin[0], 0), (self.origin[0], self.size[1]))
        pygame.draw.line(self.screen, (0, 0, 0), (0, self.origin[1]), (self.size[0], self.origin[1]))
        self.showDrawables()
        pygame.display.flip()
