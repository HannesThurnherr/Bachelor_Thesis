import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import csr_matrix, identity


def dist(a, b):
    d = np.linalg.norm(a - b)
    #d = np.sqrt(np.square(a[1] - b[1]) + np.square(a[0] - b[0]))
    return d


class nlnn:

    def __init__(self, hidden_neurons=400, input_neurons=784, output_neurons=2):
        self.coord = None
        self.hidden_neurons = hidden_neurons
        self.dim_matrix = hidden_neurons + input_neurons + output_neurons
        self.adj_matrix = np.zeros((self.dim_matrix, self.dim_matrix))
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.neuron_values = np.zeros(self.dim_matrix)

    def initialise_structure(self, connection_probability_dropoff=1.0, connection_probabily_scalar=1.0, input_connection_prob_multiplyer = 50.0, output_connection_prob_multiplyer = 30.0):
        space_size = 1
        input_n_coord = np.zeros((self.input_neurons, 2))
        output_n_coord = np.zeros((self.output_neurons, 2))
        # Initializing the positions of the neurons in the input and output layer
        for i in range(self.input_neurons):
            input_n_coord[i][1] = (space_size / 10) + space_size * 0.8 / self.input_neurons * i + 0.8 * space_size / self.input_neurons / 2
        for i in range(self.output_neurons):
            output_n_coord[i][0] = space_size
            output_n_coord[i][1] = (space_size / 10) + space_size * 0.8 / self.output_neurons * i + 0.8 * space_size / self.output_neurons / 2

        hidden_n_coord = np.zeros((self.hidden_neurons, 2))
        # Initializing the positions of the hidden neurons
        for i in range(self.hidden_neurons):
            hidden_n_coord[i] = np.random.rand(2)
            hidden_n_coord[i][0] = hidden_n_coord[i][0] * 0.8 * space_size + 0.1 * space_size
            hidden_n_coord[i][1] = hidden_n_coord[i][1] * space_size

        # initializing the connections between the neurons and their values
        connection_count = 0
        # connecting input layer to hidden neurons
        coord = np.array(list(input_n_coord) + list(hidden_n_coord) + list(output_n_coord))
        self.coord = coord
        for i in range(self.input_neurons):
            for j in range(self.input_neurons, self.input_neurons + self.hidden_neurons):
                if i != j:
                    if np.random.rand() < 1 / (dist(coord[i], coord[j]) ** connection_probability_dropoff) * connection_probabily_scalar * input_connection_prob_multiplyer:
                        self.adj_matrix[i][j] = (np.random.rand() * 4) - 2
                        connection_count += 1

        # connecting hidden neurons to eachother
        for i in range(self.input_neurons, self.input_neurons + self.hidden_neurons):
            for j in range(self.input_neurons, self.input_neurons + self.hidden_neurons):
                if i != j:
                    if np.random.rand() < 1 / (dist(coord[i], coord[j]) ** connection_probability_dropoff) * connection_probabily_scalar:
                        self.adj_matrix[i][j] = (np.random.rand() * 4) - 2
                        connection_count += 1

        # connecting hidden neurons to output layer
        for i in range(self.input_neurons, self.input_neurons + self.hidden_neurons):
            for j in range(self.input_neurons + self.hidden_neurons, self.dim_matrix):
                if i != j:
                    if np.random.rand() < 1 / (dist(coord[i], coord[j]) ** connection_probability_dropoff) * connection_probabily_scalar * output_connection_prob_multiplyer:
                        self.adj_matrix[i][j] = (np.random.rand() * 4) - 2
                        connection_count += 1
        self.adj_matrix = csr_matrix(self.adj_matrix)

    def initialise_structure_n_closest(self, hidden_neuron_connections = 5, input_neuron_connections = 10, output_neuron_connections = 10):
        space_size = 1
        input_n_coord = np.zeros((self.input_neurons, 2))
        output_n_coord = np.zeros((self.output_neurons, 2))
        # Initializing the positions of the neurons in the input and output layer
        for i in range(self.input_neurons):
            input_n_coord[i][1] = (space_size / 10) + space_size * 0.8 / self.input_neurons * i + 0.8 * space_size / self.input_neurons / 2
        for i in range(self.output_neurons):
            output_n_coord[i][0] = space_size
            output_n_coord[i][1] = (space_size / 10) + space_size * 0.8 / self.output_neurons * i + 0.8 * space_size / self.output_neurons / 2

        hidden_n_coord = np.zeros((self.hidden_neurons, 2))
        # Initializing the positions of the hidden neurons
        for i in range(self.hidden_neurons):
            hidden_n_coord[i] = np.random.rand(2)
            hidden_n_coord[i][0] = hidden_n_coord[i][0] * 0.8 * space_size + 0.1 * space_size
            hidden_n_coord[i][1] = hidden_n_coord[i][1] * space_size

        coord = np.array(list(input_n_coord) + list(hidden_n_coord) + list(output_n_coord))
        self.coord = coord

        # Compute the pairwise distances between neurons
        distances = np.linalg.norm(coord[:, np.newaxis, :] - coord[np.newaxis, :, :], axis=2)

        for i in range(self.dim_matrix):
            # Get the indices of the n closest neurons, excluding the neuron itself
            if i < self.input_neurons:
                closest_indices = np.argpartition(distances[i], input_neuron_connections + 1)[: input_neuron_connections + 1]
                closest_indices = closest_indices[closest_indices != i]
            elif i < self.input_neurons + self.hidden_neurons:
                closest_indices = np.argpartition(distances[i], hidden_neuron_connections + 1)[: hidden_neuron_connections + 1]
                closest_indices = closest_indices[closest_indices != i]
            else:
                closest_indices = np.argpartition(distances[i], output_neuron_connections + 1)[: output_neuron_connections + 1]
                closest_indices = closest_indices[closest_indices != i]

            # Connect the neuron to its n closest neighbors
            self.adj_matrix[i, closest_indices] = (np.random.rand() * 4) - 2
        self.adj_matrix = csr_matrix(self.adj_matrix)

    def display_net(self):
        self.adj_matrix = self.adj_matrix.toarray()
        plt.scatter([i[0] for i in self.coord], [i[1] for i in self.coord], s=3, color="red")
        for i in range(self.dim_matrix):
            for j in range(self.dim_matrix):
                if self.adj_matrix[j][i] != 0:
                    plt.plot([self.coord[j][0], self.coord[i][0]], [self.coord[j][1], self.coord[i][1]], linewidth=0.7,
                             color=[0, 1, 0])
        plt.scatter([i[0] for i in self.coord], [i[1] for i in self.coord], s=3, color="red")
        plt.show()
        self.adj_matrix = csr_matrix(self.adj_matrix)

    def remove_unconnected_neurons(self):
        for i in range(len(self.adj_matrix)):
            if i < len(self.adj_matrix):
                if not np.any(self.adj_matrix[i]):
                    print(i)
                    np.delete(self.coord, i)
                    np.delete(self.adj_matrix, i, 0)
                    np.delete(self.adj_matrix, i, 1)
                    if i < self.input_neurons:
                        self.input_neurons -= 1
                    elif i < self.input_neurons + self.hidden_neurons:
                        self.input_neurons -= 1
                    else:
                        self.output_neurons -= 1
                    i -= 1
                elif not np.any(self.adj_matrix.T[i]):
                    print(i)
                    np.delete(self.coord, i)
                    np.delete(self.adj_matrix, i, 0)
                    np.delete(self.adj_matrix, i, 1)
                    if i < self.input_neurons:
                        self.input_neurons -= 1
                    elif i < self.input_neurons + self.hidden_neurons:
                        self.input_neurons -= 1
                    else:
                        self.output_neurons -= 1
                    i -= 1
        self.dim_matrix = self.hidden_neurons + self.input_neurons + self.output_neurons

    def sigmoid(self, x):
        return scipy.special.expit(x) * 2 - 1

    def relu(self, x):
        return x * (x > 0)

    def leaky_relu(self, x):
        return np.where(x > 0, x, x * 0.01)

    def prop_step(self, x, activation_function = "leaky_relu"):
        #print(self.neuron_values)
        if activation_function == "relu":
            self.neuron_values = self.relu(np.dot(self.neuron_values, self.adj_matrix))
            self.neuron_values[:, :x.shape[1]] = x
        elif activation_function == "sigmoid":
            self.neuron_values = self.sigmoid(np.dot(self.neuron_values, self.adj_matrix))
            self.neuron_values[:, :x.shape[1]] = x
        elif activation_function == "leaky_relu":
            self.neuron_values = self.leaky_relu(np.dot(self.neuron_values, self.adj_matrix))
            self.neuron_values[:, :x.shape[1]] = x


    def prop_step_sparse(self, x, activation_function = "leaky_relu"):
        if activation_function == "relu":
            self.neuron_values = self.relu(self.adj_matrix.T.dot(self.neuron_values.T).T)
            self.neuron_values[:, :x.shape[1]] = x
        elif activation_function == "sigmoid":
            self.neuron_values = self.sigmoid(self.adj_matrix.T.dot(self.neuron_values.T).T)
            self.neuron_values[:, :x.shape[1]] = x
        elif activation_function == "leaky_relu":
            self.neuron_values = self.leaky_relu(self.adj_matrix.T.dot(self.neuron_values.T).T)
            self.neuron_values[:, :x.shape[1]] = x

    def get_output(self):
        return self.neuron_values[:, -self.output_neurons:]

    def predict_sparse(self, x, steps):
        self.neuron_values = np.zeros((len(x), self.dim_matrix))
        self.neuron_values[:, :x.shape[1]] = x
        for i in range(steps):
            self.prop_step_sparse(x)
        result = self.get_output()
        prediction = np.argmax(result, axis=1)
        prediction = np.eye(self.output_neurons)[prediction]
        self.neuron_values = np.zeros_like(self.neuron_values)
        return prediction, result

    def predict(self, x, steps):
        self.adj_matrix = self.adj_matrix.toarray()
        self.neuron_values = np.zeros((len(x), self.dim_matrix))
        self.neuron_values[:, :x.shape[1]] = x
        for i in range(steps):
            self.prop_step(x)
        result = self.get_output()
        prediction = np.argmax(result, axis=1)
        prediction = np.eye(self.output_neurons)[prediction]
        self.neuron_values = np.zeros_like(self.neuron_values)
        self.adj_matrix = csr_matrix(self.adj_matrix)
        return prediction, result

    def mutate_weights(self, mutation_range):
        new_matrix = self.adj_matrix.copy()
        mask = new_matrix != 0
        mutations = (np.random.rand(*new_matrix.shape) * 2 - 1) * mutation_range
        new_matrix[mask] += mutations[mask]
        return new_matrix

    def reproduce(self, amt_children, mutation_range):
        self.adj_matrix = self.adj_matrix.toarray()
        offspring = []
        for i in range(amt_children):
            new_net = nlnn(hidden_neurons=self.hidden_neurons, input_neurons=self.input_neurons, output_neurons=self.output_neurons)
            new_net.adj_matrix = self.mutate_weights(mutation_range)
            new_net.adj_matrix = csr_matrix(new_net.adj_matrix)
            new_net.coord = self.coord
            offspring.append(new_net)
        self.adj_matrix = csr_matrix(self.adj_matrix)
        return offspring

    def test_sigmoid(self):
        # Test input with one positive number
        assert self.sigmoid(np.array([1])) == 0.4621171572600098

        # Test input with one negative number
        assert self.sigmoid(np.array([-1])) == -0.4621171572600098

        # Test input with all positive numbers
        assert self.sigmoid(np.array([1, 2, 3])).all() == np.array([-0.46211716, 0.76159416, -0.90514825]).all()

        # Test input with all negative numbers
        assert self.sigmoid(np.array([-1, -2, -3])).all() == np.array([-0.46211716, -0.76159416, -0.90514825]).all()

        # Test input with a mix of positive and negative numbers
        assert self.sigmoid(np.array([-1, 2, -3])).all() == np.array([-0.23840584, 0.47681169, -0.72901441]).all()

    def test_predict(self):
        # Initialize a neural network
        n_input = 2
        n_hidden = 3
        n_output = 1
        net = nlnn(n_hidden, n_input, n_output)
        # initialize structure so the neurons have spacial coordinates
        net.initialise_structure()
        # Set the weights manually
        weights = np.array([
            [0, 0, 1.9, 0, -1.5, 0],
            [0, 0, 1.2, 0, 0, 0],
            [0, 0, 0, -1.2, 0, 0],
            [0, 0, 0, 0, 1.7, 0.5],
            [0, 0, 0, 0, 0, 1.2],
            [0, 0, 0, 0, 0, 0]
        ])
        net.adj_matrix = weights

        # Create test inputs
        x = np.array([[0.2, 0.8]])

        # Expected output
        expected_output = np.array([[-0.08909415]])
        # Test the predict function
        np.testing.assert_allclose(net.predict(x, 2)[1], expected_output, rtol=1e-5, atol=0)

    def tests(self):
        self.test_sigmoid()
        self.test_predict()


net = nlnn(hidden_neurons=100, input_neurons=3, output_neurons=3)
net.initialise_structure_n_closest(hidden_neuron_connections=10)
#net.initialise_structure(connection_probability_dropoff=3, connection_probabily_scalar=0.0003)
net.display_net()

print(net.predict(np.array([[2,1,4]]), 8), 0)
print(net.predict_sparse(np.array([[2,1,4]]), 8), 0)
