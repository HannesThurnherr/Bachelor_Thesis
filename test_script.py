import pygame
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from noise import pnoise2




# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
CLOCK = pygame.time.Clock()
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Setup screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Loss Landscape Visualization')


# Loss function with Perlin noise
def loss_function(a, b):
    b=-b
    return pnoise2(a / 10.0, b / 10.0, octaves=6, persistence=0.5, lacunarity=2.0)


# Create data for 3D plot
a_values = np.linspace(-10, 10, 100)
b_values = np.linspace(-10, 10, 100)
a_grid, b_grid = np.meshgrid(a_values, b_values)
loss_grid = np.vectorize(loss_function)(a_grid, b_grid)

min_value = np.min(loss_grid)
min_index = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
min_a = a_values[min_index[1]]
min_b = b_values[min_index[0]]



# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(min_a, min_b, min_value, color="r", s=100, label="Global Minimum")
ax.plot_surface(a_grid, b_grid, loss_grid, cmap='viridis')
ax.set_xlabel('Parameter a', fontsize=14)
ax.set_ylabel('Parameter b', fontsize=14)
ax.set_zlabel('Loss', fontsize=14)
# Hide value markers on the axes
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

ax.legend()
plt.savefig("generated_pdf_plots/loss_landscape.pdf")  # or loss_landscape.jpg for JPG
plt.show()


