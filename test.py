#!/usr/bin/env python
# coding: utf-8

# In[1]:


#TDL
"""
Optimise runtime somehow (just make it faster)
define list of options
make all the the options possible in code (implement alternative algorithms)
configure pipeline so it takes configuration as a parameter
set up grid search
???
nobel prize
"""


# In[2]:


#imports
import numpy as np
import tensorflow as tf
import tensorflow.keras
import matplotlib.pyplot as plt
import sys
from non_layered_neural_net import nlnn



# In[3]:


#Run_hyperparameters
training_run = 4
neuron_count=300
connection_probability_dropoff=3
connection_probability_scalar=0.00003
input_layer_connectivity_multiplyer=50
output_layer_connectivity_multiplyer=20
weight_initialisation_range=2
distances_from_input_output_layer_to_main_neuron_field=0.1
hidden_neuron_connections = 7
input_neuron_connections = 10
output_neuron_connections = 10
inference_steps = 8

n_closes_neurons_connection_probability="connection_prob"  #"connection_prob" /"n_closest"
activation_function="sigmoid" #relu

generation_size=5
n_survivors=3
mutation_range=0.1
training_set_size=1000
mutation_range_reducing_interval="none"
mutation_range_reducing_factor="none"

reducing_mutaiton_range="no"
stochastic_mutation_range="yes"
multiple_training_sets="yes"
keep_best_of_n_generations_keep_n_best="keep_n_best"
allow_topological_modification="no"
non_uniform_distribution_in_stochastic_mutation_range="no"


# In[37]:


#this function produces a configuration of parameters based on an input number. this allows the python script to be run with just one parameter instead of 8

def get_configuration(index):
    base_config = {
        "neuron_count": 1000,
        "connection_probability_dropoff": 3.0,
        "hidden_neuron_connections": 7,
        "inference_steps": 8,
        "n_closes_neurons_connection_probability": "connection_prob",
        "activation_function": "leaky_relu",
        "n_survivors": 3,
    }

    params_to_evaluate = [
        "neuron_count",
        "connection_probability_dropoff",
        "hidden_neuron_connections",
        "inference_steps",
        "n_closes_neurons_connection_probability",
        "activation_function",
        "n_survivors",
    ]

    param_index = index // 5
    run_index = index % 5

    param = params_to_evaluate[param_index]

    if param == "neuron_count":
        base_config[param] = int(np.linspace(200, 5000, 5)[run_index])
    elif param == "connection_probability_dropoff":
        base_config[param] = np.linspace(1, 2, 5)[run_index]
    elif param == "hidden_neuron_connections":
        base_config[param] = int(np.linspace(3, 20, 5)[run_index])
    elif param == "inference_steps":
        base_config[param] = int(np.linspace(4, 20, 5)[run_index])
    elif param == "n_closes_neurons_connection_probability":
        options = ["connection_prob", "n_closest"]
        base_config[param] = options[run_index % len(options)]
    elif param == "activation_function":
        options = ["relu", "leaky_relu"]
        base_config[param] = options[run_index % len(options)]
    elif param == "n_survivors":
        base_config[param] = int(np.linspace(1, 12, 12)[run_index])

    return base_config


# In[35]:


config = {         "training_run":training_run,
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


# In[38]:


settings = get_configuration(int(sys.argv[1]))
for i in settings.keys():
    config[i]=settings[i]


# In[5]:




# In[6]:


#load dataset
from keras.datasets import mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))


# In[7]:


#setting up conventional model 


# In[8]:





# In[9]:


#evaluate model
#accuracy = model.evaluate(mod_x_test, one_hot_y_test)
#print(accuracy)


# In[24]:


#try on all classes
#one hot encode
def one_hot_encode(x):
    out = np.zeros((len(x), max(x)+1))
    for i in range(len(x)):
        out[i][x[i]] = 1
    return out

y_train_ohe = one_hot_encode(train_y)
y_test_ohe = one_hot_encode(test_y)
#flatten images
x_train = train_X.reshape(len(train_X), 28*28)
x_test = test_X.reshape(len(test_X), 28*28)

x_test = np.array(np.array_split(x_test, 1))
y_test_ohe = np.array(np.array_split(y_test_ohe, 1))

print(x_test.shape)
print(y_test_ohe.shape)

# In[11]:





# In[12]:


#setting up evolutionary pipeline
performance_over_time = []
#creating initial population
population_size = config["generation_size"]
def create_population(population_size):
    population = []
    print("Creating networks")
    for i in range(population_size):
        net = nlnn(output_neurons = 10, hidden_neurons = config["neuron_count"])
        if config["n_closes_neurons_connection_probability"] == "connection_prob":
            net.initialise_structure(connection_probability_dropoff=config["connection_probability_dropoff"], connection_probabily_scalar=config["connection_probability_scalar"], input_connection_prob_multiplyer = config["input_layer_connectivity_multiplyer"], output_connection_prob_multiplyer = config["output_layer_connectivity_multiplyer"])
        elif config["n_closes_neurons_connection_probability"] == "n_closest":
            net.initialise_structure_n_closest(hidden_neuron_connections = config["hidden_neuron_connections"], input_neuron_connections = config["input_neuron_connections"], output_neuron_connections = config["output_neuron_connections"])
        #net.initialise_randomly()
        population.append(net)
        print("\n", end="")

    print("done!")
    return population


networks = create_population(population_size)


# In[13]:


def get_perf(t):
        return t[0]

#measure performance of all the networks
def evaluate_performance(population, x, y):
    performances = []
    print("evaluating performances", end="")
    for net in population:
        predictions = net.predict(x, config["inference_steps"])[0]
        correct_count = len(x)-(np.sum(np.abs(y - predictions))/2)
        performances.append((correct_count/len(x), net))
        #print(correct_count/len(x))
        print("|", end="")
    print(" done!", end=" ")
    #sort by best performance
    performances.sort(key = get_perf, reverse = True)
    return performances

evaluated_networks = evaluate_performance(networks, x_test[0], y_test_ohe[0])
print("best performer of this generation :", evaluated_networks[0][0])
performance_over_time.append(np.array(evaluated_networks)[:,0])


# In[14]:


from concurrent.futures import ThreadPoolExecutor

def evaluate_net(net, x, y):
    predictions = net.predict(x, config["inference_steps"])[0]
    correct_count = len(x) - (np.sum(np.abs(y - predictions)) / 2)
    performance = correct_count / len(x)
    return (performance, net)

def evaluate_performance_fast(population, x, y):
    with ThreadPoolExecutor() as executor:
        performances = list(executor.map(evaluate_net, population, [x] * len(population), [y] * len(population)))

    print("Evaluation done!")

    # Sort by best performance
    performances.sort(key=lambda x: x[0], reverse=True)
    return performances


# In[15]:


#the n best performing networks will be selected
n = config["n_survivors"]
mutation_range = config["mutation_range"]

def repopulate(evaluated_networks, mutation_range, n):
    offspring_per_network = int(population_size/n)
    next_gen = [i[1] for i in evaluated_networks[:n]]
    for net in next_gen:
        next_gen = next_gen+net.reproduce(min(offspring_per_network, population_size-(len(next_gen)+offspring_per_network)), mutation_range)
    return next_gen
    
next_generation = repopulate(evaluated_networks, mutation_range,n)


# In[16]:


performance_over_time = []


# In[17]:


test_sets_used = []
mutation_ranges = []


# In[18]:


print(np.array(performance_over_time))
for i in performance_over_time:
    print(len(i))


# In[27]:


generations = 10000
mutation_range = config["mutation_range"]/5
population_size = config["generation_size"]
test_set = 0
for gen in range(generations):
    if(config["reducing_mutaiton_range"]=="yes"):
        if(gen%config["mutation_range_reducing_interval"]==0 and gen!=0): 
            mutation_range/=config["mutation_range_reducing_factor"]
            print("decreasing mutation range from",mutation_range*config["mutation_range_reducing_factor"],"to",mutation_range)
    print("generation "+str(len(performance_over_time)+1), end=" ")

    if False and config["multiple_training_sets"] == "yes" and len(performance_over_time)%100 == 0 and gen!=0:
        test_set = np.random.randint(4)
    test_sets_used.append(test_set)
    print(" test set:",test_set,end=" ")
    evaluated_networks = evaluate_performance(next_generation, x_test[test_set], y_test_ohe[test_set])
    print(" best:", evaluated_networks[0][0], "second:", evaluated_networks[1][0], "third:", evaluated_networks[2][0])
    performance_over_time.append(list(np.array(evaluated_networks)[:,0]))
    generational_mutation_range = mutation_range
    if config["stochastic_mutation_range"]=="yes": #change back
        generational_mutation_range = np.random.rand() * mutation_range
    mutation_ranges.append(generational_mutation_range)
    print("mutating in range:", generational_mutation_range)
    next_generation = repopulate(evaluated_networks, generational_mutation_range, config["n_survivors"])
    if(gen%10==0):
        """
        plt.plot(performance_over_time, alpha= 0.1)
        plt.plot(np.array(performance_over_time)[:,0])
        plt.show()
        """
        print("average best of last 100 generations",np.average(np.array(performance_over_time)[-100:,0]))
    


# In[29]:


#visualising performance across the 10 different sets

performance_hist = np.array(performance_over_time)[-len(test_sets_used):,0]
set_perf = []
for i in range(10):
    set_perf.append([])
    


for i in range(len(test_sets_used)-10):
    set_perf[test_sets_used[i]].append(performance_hist[i])

for i in set_perf:
    plt.plot(i, alpha=0.8)
plt.show()
"""
plt.violinplot(set_perf)
plt.show()
"""
performance_hist = np.array(performance_over_time)[-len(test_sets_used):]
performance_changes = []
for i in range(len(performance_hist)-1):
    performance_changes.append(-(np.average(performance_hist[i])-np.average(performance_hist[i+1])))


plt.scatter(np.array(performance_changes)[:len(mutation_ranges)], np.array(mutation_ranges)[:len(performance_changes)])
plt.xlabel('change in perform')
plt.ylabel('Y-axis label')
plt.show()
plt.violinplot(np.array(mutation_ranges))
plt.show()


# In[30]:


#TQDM -> loading bars

plt.plot(performance_over_time, alpha= 0.1)
plt.plot(np.array(performance_over_time)[:,0])
#plt.plot(list(0.99*np.ones(len(performance_over_time))))
plt.show()


# In[31]:


plt.plot(np.average(np.split(np.array(performance_over_time)[:6400,0], 40),axis =1))


# In[32]:


log = {"training_run":training_run,
       "neuron_count":neuron_count,
       "connection_probability_dropoff":connection_probability_dropoff,
       "connection_probability_scalar":connection_probability_scalar,
       "input_layer_connectivity_multiplyer":input_layer_connectivity_multiplyer,
       "output_layer_connectivity_multiplyer":output_layer_connectivity_multiplyer,
       "weight_initialisation_range":weight_initialisation_range,
       "distances_from_input_output_layer_to_main_neuron_field":distances_from_input_output_layer_to_main_neuron_field,
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
       "keep_best_of_n_generations_keep_n_best":keep_best_of_n_generations_keep_n_best,
       "allow_topological_modification":allow_topological_modification,
       "non-uniform_distribution_in_stochastic_mutation_range":non_uniform_distribution_in_stochastic_mutation_range }


# In[33]:


np.savetxt('training_run_'+str(training_run)+'_performance.csv', performance_over_time, delimiter=',')
np.savetxt('training_run_'+str(training_run)+'_test_sets_used.csv', test_sets_used, delimiter=',')
np.savetxt('training_run_'+str(training_run)+'_mutation_ranges.csv', mutation_ranges, delimiter=',')
np.save('training_run_'+str(training_run)+'_config.npy', np.array(log))


# In[34]:


#Ideas to improve GA
#Cap unused neurons (if collumn or line of adj matrix empty, kill both) not worth it?
#topology modification


# In[ ]:





# In[ ]:




