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

    def showTime(self, screen):
        myfont = pygame.font.SysFont("monospace", 11)
        label = myfont.render(f"{self.simulation.getRunningTime()}s", 1, (0, 0, 0))
        screen.blit(label, (0,0))

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
                c = 10
                p = (float(self.origin[0] + particle.x[0]), float(self.origin[1] + particle.x[1]))
                a = (float(p[0] + particle.a[0]*c), float(p[1] + particle.a[1]*c))
                aApplied = (float(p[0] + particle.aApplied[0]*c), float(p[1] + particle.aApplied[1]*c))
                aConstraint = (float(p[0] + particle.aConstraint[0]*c), float(p[1] + particle.aConstraint[1]*c))
                pygame.draw.line(self.screen, (0, 0, 0), p, a)
                pygame.draw.line(self.screen, (0, 255, 0), p, aApplied)
                pygame.draw.line(self.screen, (255, 0, 0), p, aConstraint)
                pygame.draw.circle(self.screen, (0, 0, 255), p, 5)
            self.showTime(self.screen)
            pygame.display.flip()

            time.sleep(float(self.timestep))

        pygame.quit()

