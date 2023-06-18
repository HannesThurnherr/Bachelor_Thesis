import numpy as np

class lnn:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        for i in range(len(self.layers)-1):
            self.weights.append(np.random.rand(layers[i],layers[i+1])*4-2)

    def relu(self, x):
        return x * (x > 0)

    def leaky_relu(self, x):
        return np.where(x > 0, x, x * 0.01)

    def predict(self, x, activation_function = "leaky_relu"):
        for i in range(len(self.weights)):
            if activation_function=="leaky_relu":
                x = self.leaky_relu(x @ self.weights[i])
            elif activation_function == "relu":
                x = self.relu(x @ self.weights[i])
        result = x
        prediction = np.argmax(result, axis=1)
        prediction = np.eye(self.layers[-1])[prediction]
        return prediction, result

    def mutate_weights(self, mutation_range):
        new_weights = []
        for i in range(len(self.weights)):
            new_matrix = self.weights[i].copy()
            mutations = (np.random.rand(*new_matrix.shape) * 2 - 1) * mutation_range
            new_weights.append(new_matrix + mutations)
        return new_weights

    def reproduce(self, amt_children, mutation_range):
        offspring = []
        for i in range(amt_children):
            new_net = lnn(self.layers)
            new_net.weights = self.mutate_weights(mutation_range)
            offspring.append(new_net)
        return offspring