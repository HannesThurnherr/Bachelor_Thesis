import numpy as np


def generate_run(run, default = False):
    training_run = run
    neuron_count = 1000 #min: 200, max 5000
    connection_probability_dropoff = 3.0 #min: 1, max 2
    connection_probability_scalar = 0.00003 #between 0.00005 and 0.0000005
    input_layer_connectivity_multiplyer = 50.0 #between 1.0 and 100.0
    output_layer_connectivity_multiplyer = 20.0 #between 1.0 and 100.0
    weight_initialisation_range = 2 #keep it at 2
    hidden_neuron_connections = 7 #between 3 and 20
    input_neuron_connections = 10 #between 5 and 30
    output_neuron_connections = 10 #between 5 and 30
    inference_steps = 8 #between 4 and 20

    n_closes_neurons_connection_probability = "connection_prob"  #alternative: "n_closest"
    activation_function = "leaky_relu"  #alteratives: relu, leaky_relu

    generation_size = 12
    n_survivors = 3 #integer between 1 and the generation_size
    mutation_range = 0.1 # a random value between 0 and 5 biased towards small values
    training_set_size = 1000 #keep this at 1000
    mutation_range_reducing_interval = "none" #this value should only be returned when the reducing_mutation_range variable is "yes"
    mutation_range_reducing_factor = "none" #this value should only be returned when the reducing_mutation_range variable is "yes"

    reducing_mutaiton_range = "no" #"yes"
    stochastic_mutation_range = "yes" #"no"
    multiple_training_sets = "yes" #"no"
    allow_topological_modification = "no" #"yes"

    if default:
        return {   "training_run":training_run,
                   "neuron_count":neuron_count,
                   "connection_probability_dropoff":connection_probability_dropoff,
                   "connection_probability_scalar":connection_probability_scalar,
                   "input_layer_connectivity_multiplyer":input_layer_connectivity_multiplyer,
                   "output_layer_connectivity_multiplyer":output_layer_connectivity_multiplyer,
                   "weight_initialisation_range":weight_initialisation_range,
                   "n_closes_neurons_connection_probability":n_closes_neurons_connection_probability,
                   "hidden_neuron_connections" : hidden_neuron_connections,
                   "input_neuron_connections" : input_neuron_connections,
                   "output_neuron_connections" : output_neuron_connections,
                   "inference_steps" : inference_steps,
                   "activation_function":activation_function,
                   "generation_size":generation_size,
                   "n_survivors":n_survivors,
                   "mutation_range":mutation_range,
                   "training_set_size":training_set_size,
                   "mutation_range_reducing_interval":mutation_range_reducing_interval,
                   "mutation_range_reducing_factor":mutation_range_reducing_factor,
                   "reducing_mutaiton_range":reducing_mutaiton_range,
                   "stochastic_mutation_range":stochastic_mutation_range,
                   "multiple_training_sets":multiple_training_sets,
                   "allow_topological_modification":allow_topological_modification
                }

    config = {
        "training_run": training_run,
        "neuron_count": np.random.randint(200, 5001),
        "connection_probability_dropoff": np.random.uniform(1, 2),
        "connection_probability_scalar": np.random.uniform(0.0000005, 0.00005),
        "input_layer_connectivity_multiplyer": np.random.uniform(1.0, 100.0),
        "output_layer_connectivity_multiplyer": np.random.uniform(1.0, 100.0),
        "weight_initialisation_range": 2,
        "hidden_neuron_connections": np.random.randint(3, 21),
        "input_neuron_connections": np.random.randint(5, 31),
        "output_neuron_connections": np.random.randint(5, 31),
        "inference_steps": np.random.randint(4, 21),
        "n_closes_neurons_connection_probability": np.random.choice(["connection_prob", "n_closest"]),
        "activation_function": "leaky_relu",#np.random.choice(["sigmoid", "relu", "leaky_relu"]),
        "generation_size": 12,
        "n_survivors": np.random.randint(1, 13),
        "training_set_size": 1000,
        "reducing_mutaiton_range": np.random.choice(["yes", "no"]),
        "stochastic_mutation_range": np.random.choice(["yes", "no"]),
        "multiple_training_sets": np.random.choice(["yes", "no"]),
        "allow_topological_modification": np.random.choice(["yes", "no"]),
        "mutation_range" : np.random.uniform(0, 5),
        "mutation_range_reducing_interval" : np.random.randint(5, 100),
        "mutation_range_reducing_factor" : np.random.uniform(0.1, 1)
    }

    return config


n = generate_run(2)

for i in list(n.keys()):
    print(i,":",n[i])