import numpy as np
import pygame


class Particle:
    i: int
    x: np.ndarray
    v: np.ndarray
    a: np.ndarray
    aApplied: np.ndarray
    aConstraint: np.ndarray

    def __init__(self, i: int, x: np.ndarray, v: np.ndarray):
        self.i, self.x, self.v = i, x, v
        self.a, self.aApplied, self.aConstraint = np.zeros_like(v), np.zeros_like(v), np.zeros_like(v)

    def draw(self, surface: pygame.Surface, origin: np.ndarray):
        c = 100
        p = origin + self.x
        a = p + self.a * c
        aApplied = p + self.aApplied * c
        aConstraint = p + self.aConstraint * c
        pygame.draw.line(surface, (0, 0, 0), p, a)
        pygame.draw.line(surface, (0, 255, 0), p, aApplied)
        pygame.draw.line(surface, (255, 0, 0), p, aConstraint)
        pygame.draw.circle(surface, (0, 0, 255), p, 5)
