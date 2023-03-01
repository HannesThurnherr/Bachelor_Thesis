import numpy as np
import matplotlib.pyplot as plt
import scipy
import unittest
"""
TDL:
make mutation:
    mutate_weight function
    make_offspring_function

tests:
    
"""

def dist(a,b):
    #   d = np.linalg.norm(np.array[a,b])
    d = np.sqrt(np.square(a[1]-b[1])+np.square(a[0]-b[0]))
    return d

class nlnn:

    def __init__(self, hidden_neurons = 400, input_neurons = 784, output_neurons = 2):
        self.hidden_neurons = hidden_neurons
        self.dim_matrix = hidden_neurons + input_neurons + output_neurons
        self.adj_matrix = np.zeros((self.dim_matrix, self.dim_matrix))
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.neuron_values = np.zeros(self.dim_matrix)


    def initialise_structure(self, connection_probability_dropoff = 1, connection_probabily_scalar = 1):
        space_size = 1
        input_n_coord = np.zeros((self.input_neurons,2))
        output_n_coord = np.zeros((self.output_neurons, 2))
        #Initializing the positions of the neurons in the input and output layer
        for i in range(self.input_neurons):
            input_n_coord[i][1] = (space_size/10)+space_size*0.8/self.input_neurons*i+0.8*space_size/self.input_neurons/2
        for i in range(self.output_neurons):
            output_n_coord[i][0] = space_size
            output_n_coord[i][1] = (space_size/10)+space_size*0.8/self.output_neurons*i+0.8*space_size/self.output_neurons/2

        hidden_n_coord = np.zeros((self.hidden_neurons,2))
        # Initializing the positions of the hidden neurons
        for i in range(self.hidden_neurons):
            hidden_n_coord[i] = np.random.rand(2)
            hidden_n_coord[i][0] = hidden_n_coord[i][0] * 0.8 * space_size + 0.1 * space_size
            hidden_n_coord[i][1] = hidden_n_coord[i][1] * space_size

        #initializing the connections between the neurons and their values
        connection_count = 0
        #connecting input layer to hidden neurons
        coord = np.array(list(input_n_coord)+list(hidden_n_coord)+list(output_n_coord))
        self.coord = coord
        for i in range(self.input_neurons):
            for j in range(self.input_neurons,self.input_neurons + self.hidden_neurons):
                if i != j:
                    if np.random.rand() < 1 / (dist(coord[i] , coord[j]) ** connection_probability_dropoff) * connection_probabily_scalar*50:
                        self.adj_matrix[i][j] = (np.random.rand() * 4) - 2 #weight initialisation in range (-4,4) based on https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network#:~:text=I%20have%20just%20heard%2C%20that,inputs%20to%20a%20given%20neuron.
                        connection_count+=1

        #connecting hidden neurons to eachother
        for i in range(self.dim_matrix):
            for j in range(self.dim_matrix):
                if i != j:
                    if np.random.rand() < 1 / (dist(coord[i], coord[j]) ** connection_probability_dropoff) * connection_probabily_scalar:
                        self.adj_matrix[i][j] = (np.random.rand() * 4) - 2   # weight initialisation in range (-4,4) based on https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network#:~:text=I%20have%20just%20heard%2C%20that,inputs%20to%20a%20given%20neuron.
                        connection_count += 1

        #connecting hidden neurons to output layer
        for i in range(self.input_neurons,self.input_neurons + self.hidden_neurons):
            for j in range(self.input_neurons+ self.hidden_neurons, self.dim_matrix):
                if i != j:
                    if np.random.rand() < 1 / (dist(coord[i], coord[j]) ** connection_probability_dropoff) * connection_probabily_scalar * 20:
                        self.adj_matrix[i][j] = (np.random.rand() * 4) - 2   # weight initialisation in range (-4,4) based on https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network#:~:text=I%20have%20just%20heard%2C%20that,inputs%20to%20a%20given%20neuron.
                        connection_count += 1


        #self.display_net()


    def display_net(self):
        for i in range(self.dim_matrix):
            for j in range(self.dim_matrix):
                if self.adj_matrix[j][i]!=0:
                    plt.plot([self.coord[j][0],self.coord[i][0]],[self.coord[j][1],self.coord[i][1]],linewidth = 0.7 , color= [0, 1 * (1 - abs(self.adj_matrix[j][i]) / 4), 0.2 * (abs(self.adj_matrix[j][i]) / 4)])
        plt.scatter([i[0] for i in self.coord], [i[1] for i in self.coord],  s = 3, color = "red")
        plt.show()

    def sigmoid(self, vector):
        return scipy.special.expit(vector)*2-1

    def prop_step(self, x):
        self.neuron_values = self.sigmoid(np.dot(self.adj_matrix.T, self.neuron_values))
        self.neuron_values[:len(x)] = x

    def get_output(self):
        return self.neuron_values[-self.output_neurons:]


    def predict(self, x, steps):
        assert x.shape == (self.input_neurons,) , ("input does not have the right shape ", x.shape, " vs. ", self.input_neurons)
        self.neuron_values[:len(x)] = x
        for i in range(steps):
            self.prop_step(x)
        result = self.get_output()
        prediction = np.zeros(self.output_neurons)
        prediction[np.argmax(result)]=1
        self.neuron_values = np.zeros_like(self.neuron_values)
        return prediction,result

    def mutate_weights(self, mutation_range):
        new_matrix = np.zeros(self.adj_matrix.shape)
        for i in range(len(self.adj_matrix)):
            for j in range(len(self.adj_matrix[i])):
                if self.adj_matrix[i][j]!=0:
                    new_matrix[i][j] = self.adj_matrix[i][j]+(np.random.rand()*2-1)*mutation_range
        
        return new_matrix

    def reproduce(self, amt_children, mutation_range):
        offspring = []
        for i in range(amt_children):
            new_net = nlnn(hidden_neurons= self.hidden_neurons, input_neurons = self.input_neurons, output_neurons = self.output_neurons)
            new_net.adj_matrix = self.mutate_weights(mutation_range)
            offspring.append(new_net)
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
        #initialize structure so the neurons have spacial coordinates
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
        x = np.array([0.2, 0.8])

        # Expected output
        expected_output = np.array([-0.08909415])
        # Test the predict function
        np.testing.assert_allclose(net.predict(x, 2)[1], expected_output, rtol=1e-5, atol=0)

    def tests(self):
        self.test_sigmoid()
        self.test_predict()



net = nlnn(hidden_neurons= 1000, input_neurons = 3, output_neurons = 2)
net.initialise_structure(connection_probability_dropoff=3, connection_probabily_scalar=0.00003)

#for i in range(10):
    #print(net.predict(np.array([0.2, 0.1, -0.3]), i))

net.tests()

print(net.mutate_weights(0.1))
