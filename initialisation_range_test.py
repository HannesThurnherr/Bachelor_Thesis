import numpy as np
import matplotlib.pyplot as plt

for j in range(10):
    lst = [np.random.rand()*2-1]
    for i in range(1000):
        lst.append(lst[-1]+np.random.rand()*2-1)

    plt.plot(lst)
    plt.show()