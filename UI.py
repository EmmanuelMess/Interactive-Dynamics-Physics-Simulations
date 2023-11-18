from itertools import count

import pygame
import numpy as np
import time
from typing import List

from Constraint import Constraint
from Particle import Particle
from Simulation import Simulation


class UI:
    def __init__(self, particles: List[Particle], constraints: List[Constraint], simulation: Simulation, timestep: np.float64):
        self.particles, self.constraints, self.simulation, self.timestep = particles, constraints, simulation, timestep
        self.size = [500, 500]
        self.origin = np.array(self.size)/2
        self.running = True

        np.set_printoptions(suppress=True)

        pygame.init()
        self.screen = pygame.display.set_mode(self.size)
        self.font = pygame.font.SysFont("monospace", 11)

    def showTime(self):
        label = self.font.render(f"t {self.simulation.getRunningTime()}s", 1, (0, 0, 0))
        self.screen.blit(label, (0, 0))

    def showParticles(self):
        for yPositionParticle, particle in zip(count(10, 40), self.particles):
            label = self.font.render(f"p {particle.i}", 1, (0, 0, 0))
            self.screen.blit(label, (0, yPositionParticle))

            for yPositionValues, string in zip(count(yPositionParticle+10, 10), [f"x {particle.x}", f"v {particle.v}", f"a {particle.a}"]):
                label = self.font.render(string, 1, (0, 0, 0))
                self.screen.blit(label, (10, yPositionValues))

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.simulation.update()

            self.screen.fill((255, 255, 255))
            for constraint in self.constraints:
                constraint.surface(self.screen, self.origin)
            pygame.draw.line(self.screen, (0, 0, 0), (self.origin[0], 0), (self.origin[0], self.size[1]))
            pygame.draw.line(self.screen, (0, 0, 0), (0, self.origin[1]), (self.size[0], self.origin[1]))
            for particle in self.particles:
                particle.draw(self.screen, self.origin)
            self.showTime()
            self.showParticles()
            pygame.display.flip()

        pygame.quit()

