import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from non_layered_neural_net import nlnn

neurons = 25
input_n = 4
output_n = 4

net = nlnn(input_neurons=4, hidden_neurons=neurons-input_n-output_n, output_neurons=2)
net.initialise_structure_n_closest()
#net.display_net()

layers = 5

x = np.tile(net.coord.T[0], layers)
y = np.tile(net.coord.T[1], layers)
z = np.concatenate([np.repeat(i, len(net.coord.T[0])) for i in range(layers)])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

def update(num):
    ax.view_init(30, num)

ani = FuncAnimation(fig, update, frames=range(0,360), interval=50)

ax.scatter(z, x, y)
for layer in range(layers-1):
    for i in range(net.dim_matrix):
        for j in range(net.dim_matrix):
            if(net.adj_matrix.toarray()[i][j]!=0):
                ax.plot([layer, layer + 1], [x[i], x[j]], [y[i], y[j]], color="green", linewidth=0.5, alpha=0.2)

plt.show()

adj_matrix = np.random.rand(neurons,neurons)
coord = np.array([(i, j) for i in range(5) for j in range(5)])

x = np.tile(coord.T[0], layers)
y = np.tile(coord.T[1], layers)
z = np.concatenate([np.repeat(i, len(coord.T[0])) for i in range(layers)])

fig = plt.figure()

ax = fig.add_subplot(projection='3d')

ani = FuncAnimation(fig, update, frames=range(0,360), interval=50)

ax.scatter(z, x, y)
for layer in range(layers-1):
    for i in range(neurons):
        for j in range(neurons):
            if(adj_matrix[i][j]!=0):
                ax.plot([layer, layer + 1], [x[i], x[j]], [y[i], y[j]], color="green", linewidth=0.5, alpha=0.2)

plt.show()
