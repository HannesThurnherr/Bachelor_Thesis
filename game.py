import pygame
import math
import random
import sys
import math
import noise

def custom_map(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def perlin_function(x, y):
    angle = custom_map(noise.snoise3(x / 300, y / 300, 0), 0, 1, 0, 2 * math.pi)
    raw_length = noise.snoise3(x / 300, y / 300, 1000)

    # Custom sigmoid-like function for length
    k = -10   # Adjust this value to control the sharpness of the transition between short and long vectors
    sigmoid_length = 1 / (1 + math.exp(-k * (raw_length - 0.5)))

    # Map the sigmoid output to the desired length range
    min_length = 0.5
    max_length = 8
    length = custom_map(sigmoid_length, 0, 1, min_length, max_length)

    # Perlin
    out = [x - math.sin(angle) * length, y - math.cos(angle) * length]
    return out

pygame.init()
(width, height) = (1730, 1050)
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

class LineDraw:
    def __init__(self, posx, posy, eq):
        self.posx = posx
        self.posy = posy
        self.eq = eq
        self.arrowNotDrawn = False
        self.colorshift = 80

    def update(self):
        futurePos = self.eq.eq(self.posx, self.posy)
        self.posx = futurePos[0]
        self.posy = futurePos[1]

    def show(self):
        futurePos = self.eq.eq(self.posx, self.posy)
        vLength = math.sqrt((futurePos[0] - self.posx) ** 2 + (futurePos[1] - self.posy) ** 2)

        r = int(min(max(vLength * self.colorshift, 0), 255))
        g = int(min(max(255 - (vLength * self.colorshift / 3), 0), 255))
        b = int(min(max(255 - vLength * self.colorshift, 0), 255))

        color = (r, g, b)

        pygame.draw.line(screen, color, (self.posx + width / 2, height / 2 - self.posy),
                         (futurePos[0] + width / 2, height / 2 - futurePos[1]), 1)


class DiffEq:
    def eq(self, x, y):
        # circle
        #out = [x + (y * 0.006 * math.sin(pygame.time.get_ticks() / 8000.0)), y + (x * 0.006 * math.sin(pygame.time.get_ticks() / 4000.0))]
        out = perlin_function(x, y)
        return out


def main():
    global screen, width, height
    pygame.init()

    width = 1730
    height = 1050
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Differential Equation Visualization")
    clock = pygame.time.Clock()
    running = True

    lineDraws = [LineDraw(random.uniform(-width / 2, width / 2), random.uniform(-height / 2, height / 2), DiffEq()) for _ in range(120 * 120)]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Draw a semi-transparent black surface on top of the existing content
        trail_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        trail_surface.fill((0, 0, 0, 3))
        screen.blit(trail_surface, (0, 0))

        for ld in lineDraws:
            ld.show()
            ld.update()

            # Check if the LineDraw object is offscreen
            if ld.posx < -width / 2 or ld.posx > width / 2 or ld.posy > height / 2 or ld.posy < -height / 2:
                ld.posx = random.uniform(-width / 2, width / 2)
                ld.posy = random.uniform(-height / 2, height / 2)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
