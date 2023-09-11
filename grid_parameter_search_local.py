print("script started")

import numpy as np
import sys
import os
from non_layered_neural_net import nlnn

sys.argv.append(2)
training_run = int(sys.argv[1])
neuron_count = 300
connection_probability_dropoff = 3
connection_probability_scalar = 0.00003
input_layer_connectivity_multiplyer = 50
output_layer_connectivity_multiplyer = 20
weight_initialisation_range = 2
distances_from_input_output_layer_to_main_neuron_field = 0.1
hidden_neuron_connections = 7
input_neuron_connections = 10
output_neuron_connections = 10
inference_steps = 8
n_closes_neurons_connection_probability = "connection_prob"  # "connection_prob" /"n_closest"
activation_function = "sigmoid"  # relu
generation_size = 15
n_survivors = 3
mutation_range = 0.1
training_set_size = 1000  # maybe make 10000?
mutation_range_reducing_interval = "none"
mutation_range_reducing_factor = "none"
reducing_mutaiton_range = "no"
stochastic_mutation_range = "yes"
multiple_training_sets = "yes"
keep_best_of_n_generations_keep_n_best = "keep_n_best"
allow_topological_modification = "no"
non_uniform_distribution_in_stochastic_mutation_range = "no"


# this function produces a configuration of parameters based on an input number. this allows the python script to be run with just one parameter instead of 8
def get_configuration(index):
    neuron_counts = np.round(np.linspace(9, 19, 5) ** 2.5).astype(int)
    connection_probability_dropoffs = np.linspace(1, 4, 5)
    inference_steps = np.round(np.linspace(6, 20, 5)).astype(int)
    n_survivors = np.round(np.linspace(1, 12, 5)).astype(int)
    activation_functions = ["leaky_relu", "relu"]

    configurations = []

    for activation_function in activation_functions:
        for connection_probability_dropoff in connection_probability_dropoffs:
            for inference_step in inference_steps:
                for neuron_count in neuron_counts:
                    for n_survivor in n_survivors:
                        configuration = {
                            "neuron_count": neuron_count,
                            "connection_probability_dropoff": connection_probability_dropoff,
                            "hidden_neuron_connections": 6,
                            "inference_steps": inference_step,
                            "n_survivors": n_survivor,
                            "activation_function": activation_function,
                        }
                        configurations.append(configuration)

    return np.random.choice(configurations)


config = {"training_run": training_run,
          "neuron_count": neuron_count,
          "connection_probability_dropoff": connection_probability_dropoff,
          "connection_probability_scalar": connection_probability_scalar,
          "input_layer_connectivity_multiplyer": input_layer_connectivity_multiplyer,
          "output_layer_connectivity_multiplyer": output_layer_connectivity_multiplyer,
          "weight_initialisation_range": weight_initialisation_range,
          "n_closes_neurons_connection_probability": n_closes_neurons_connection_probability,
          "hidden_neuron_connections": hidden_neuron_connections,
          "input_neuron_connections": input_neuron_connections,
          "output_neuron_connections": output_neuron_connections,
          "inference_steps": inference_steps,
          "activation_function": activation_function,
          "generation_size": generation_size,
          "n_survivors": n_survivors,
          "mutation_range": mutation_range,
          "training_set_size": training_set_size,
          "mutation_range_reducing_interval": mutation_range_reducing_interval,
          "mutation_range_reducing_factor": mutation_range_reducing_factor,
          "reducing_mutaiton_range": reducing_mutaiton_range,
          "stochastic_mutation_range": stochastic_mutation_range,
          "multiple_training_sets": multiple_training_sets,
          "allow_topological_modification": allow_topological_modification
          }


def load_local_mnist_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)


def one_hot_encode(x):
    out = np.zeros((len(x), max(x) + 1))
    for i in range(len(x)):
        out[i][x[i]] = 1
    return out


def create_population(population_size):
    population = []
    print("Creating networks")
    for i in range(population_size):
        net = nlnn(output_neurons=10, hidden_neurons=config["neuron_count"])
        if config["n_closes_neurons_connection_probability"] == "connection_prob":
            net.initialise_structure(connection_probability_dropoff=config["connection_probability_dropoff"], connection_probabily_scalar=config["connection_probability_scalar"],
                                     input_connection_prob_multiplyer=config["input_layer_connectivity_multiplyer"], output_connection_prob_multiplyer=config["output_layer_connectivity_multiplyer"])
        elif config["n_closes_neurons_connection_probability"] == "n_closest":
            net.initialise_structure_n_closest(hidden_neuron_connections=config["hidden_neuron_connections"], input_neuron_connections=config["input_neuron_connections"],
                                               output_neuron_connections=config["output_neuron_connections"])
        # net.initialise_randomly()
        population.append(net)
        print("|", end="")

    print("done!")
    return population


def get_perf(t):
    return t[0]


def evaluate_performance(population, x, y):
    performances = []
    print("evaluating performances", end="")
    for net in population:
        predictions = net.predict(x, config["inference_steps"])[0]
        correct_count = len(x) - (np.sum(np.abs(y - predictions)) / 2)
        performances.append((correct_count / len(x), net))
        print("|", end="")
    print(" done!", end=" ")
    # sort by best performance
    performances.sort(key=get_perf, reverse=True)
    return performances


def repopulate(evaluated_networks, mutation_range, n):
    offspring_per_network = int(population_size / n)
    parents = [i[1] for i in evaluated_networks[:n]]
    offspring = []
    for net in parents:
        net_offspring = net.reproduce(min(offspring_per_network, population_size - len(offspring)), mutation_range)
        offspring.extend(net_offspring)
    next_gen = parents + offspring
    next_gen = next_gen[:population_size]
    return next_gen


settings = get_configuration(int(sys.argv[1]))
for i in settings.keys():
    config[i] = settings[i]
for i in config.keys():
    print(i, ":", config[i])

(train_X, train_y), (test_X, test_y) = load_local_mnist_data('mnist.npz')

y_train_ohe = one_hot_encode(train_y)
y_test_ohe = one_hot_encode(test_y)
# flatten images
x_train = train_X.reshape(len(train_X), 28 * 28)
x_test = test_X.reshape(len(test_X), 28 * 28)

x_test = np.array_split(x_test, 10)
y_test_ohe = np.array_split(y_test_ohe, 10)

performance_over_time = []
test_sets_used = []
mutation_ranges = []

# Training loop
# loading hyperparameters
n = config["n_survivors"]
mutation_range = config["mutation_range"]
population_size = config["generation_size"]
print_graphs = False

networks = create_population(population_size)
networks = evaluate_performance(networks, x_test[0], y_test_ohe[0])
print("best performer of this generation :", networks[0][0])
performance_over_time.append(np.array(networks)[:, 0])
networks = repopulate(networks, mutation_range, n)

generations = 1000
test_set = 0
for gen in range(generations):
    if config["reducing_mutaiton_range"] == "yes":
        if gen % config["mutation_range_reducing_interval"] == 0 and gen != 0:
            mutation_range /= config["mutation_range_reducing_factor"]
            print("decreasing mutation range from", mutation_range * config["mutation_range_reducing_factor"], "to", mutation_range)
    print("generation " + str(len(performance_over_time) + 1), end=" ")

    if config["multiple_training_sets"] == "yes" and gen != 0:
        test_set = np.random.randint(10)
    test_sets_used.append(test_set)
    print(" test set:", test_set, end=" ")
    networks = evaluate_performance(networks, x_test[test_set], y_test_ohe[test_set])

    print(" best:", networks[0][0])  # , "second:", evaluated_networks[1][0], "third:", evaluated_networks[2][0])
    performance_over_time.append(np.array(networks)[:, 0])
    generational_mutation_range = mutation_range
    if config["stochastic_mutation_range"] == "yes":  # change back
        generational_mutation_range = np.random.rand() * mutation_range
    mutation_ranges.append(generational_mutation_range)
    print("mutating in range:", generational_mutation_range)

    next_gen = repopulate(networks, generational_mutation_range, config["n_survivors"])

    del networks
    networks = next_gen
    if (gen % 10 == 0) and print_graphs:
        print("average best of last 100 generations", np.average(np.array(performance_over_time)[-100:, 0]))

folder_name = 'run_' + str(training_run)
suffix = 1

while os.path.exists(folder_name):
    suffix += 1
    folder_name = f"{folder_name}_{suffix}"

os.makedirs(folder_name)

print("saving files")

performance_over_time_array = np.array(performance_over_time, dtype=np.float64)

np.savetxt(folder_name + '/training_run_' + str(training_run) + '_performance.csv', performance_over_time_array, delimiter=',')
print("performances saved successfully")
np.savetxt(folder_name + '/training_run_' + str(training_run) + '_test_sets_used.csv', test_sets_used, delimiter=',')
print("test sets saved successfully")
np.savetxt(folder_name + '/training_run_' + str(training_run) + '_mutation_ranges.csv', mutation_ranges, delimiter=',')
print("mutation ranges saved successfully")
np.save(folder_name + '/training_run_' + str(training_run) + '_config.npy', np.array(config))
print("config saved successfully \n COMPLETED RUN SCCESSFULLY")
