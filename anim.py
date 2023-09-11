import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pickle

# Load the data
with open('performance_over_time_nlnn.pkl', 'rb') as f:
    data = pickle.load(f)

# Extracting the best-performing member's accuracy for each generation
best_performance_per_generation = [np.max(gen) for gen in data]

# Initialize the plot for animation
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2, color='green')
ax.set_ylim(min(best_performance_per_generation), max(best_performance_per_generation))
ax.set_xlim(0, len(best_performance_per_generation))
ax.set_xlabel('Generation')
ax.set_ylabel('Accuracy')
plt.title('Best Performance Over Generations')

# Initialize the data for the animation
def init():
    line.set_data([], [])
    return line,

# Update the data for each frame
def update(frame):
    x = np.arange(frame + 1)
    y = best_performance_per_generation[:frame + 1]
    line.set_data(x, y)
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(best_performance_per_generation), init_func=init, blit=True)

# Save the animation
ani.save('best_performance_over_generations.mp4', writer='ffmpeg', fps=20)

# Or display the animation if you prefer
# plt.show()
