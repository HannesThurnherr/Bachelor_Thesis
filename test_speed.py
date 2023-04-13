import numpy as np
import time


def speedtest(dim):
    start = time.time()



    matrix = np.random.rand(dim,dim)
    coord = np.random.rand(dim,2)

    def dist(a, b):
        d = np.linalg.norm(a - b)
        #d = np.sqrt(np.square(a[1] - b[1]) + np.square(a[0] - b[0]))
        return d

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if i != j:
                if np.random.rand() < 1 / (dist(coord[i], coord[j]) ** 3) * 0.0000000023:
                    matrix[i][j] = (np.random.rand() * 4) - 2


    print("time:",time.time()-start)

