import numpy as np

def get_configuration(index):
    base_config = {
        "neuron_count": 1000,
        "connection_probability_dropoff": 3.0,
        "hidden_neuron_connections": 6,
        "inference_steps": 8,
        "n_survivors": 3,
        "activation_function": "leaky_relu",
    }

    params_to_evaluate = list(base_config.keys())
    param_index = index // 5
    run_index = index % 5

    param = params_to_evaluate[param_index]

    if param == "neuron_count":
        base_config[param] = int((np.linspace(9, 19, 5) ** 2.5)[run_index])
    elif param == "connection_probability_dropoff":
        base_config[param] = np.linspace(1, 4, 5)[run_index]
    elif param == "hidden_neuron_connections":
        base_config[param] = int(np.linspace(3, 20, 5)[run_index])
        base_config["n_closes_neurons_connection_probability"] = "n_closest"
    elif param == "inference_steps":
        base_config[param] = int(np.linspace(6, 20, 5)[run_index])
    elif param == "n_survivors":
        base_config[param] = int(np.linspace(1, 15, 5)[run_index])
    elif param == "activation_function":
        options = ["relu", "leaky_relu"]
        base_config[param] = options[run_index % len(options)]

    return base_config


def get_modified_param_string(modified_config):
    default_config = {
        "neuron_count": 1000,
        "connection_probability_dropoff": 3.0,
        "hidden_neuron_connections": 6,
        "inference_steps": 8,
        "activation_function": "leaky_relu",
        "n_survivors": 3,
    }

    modified_param = "neuron_count"
    for i in default_config.keys():
        if modified_config[i] != default_config[i]:
            #print(i, modified_config[i], default_config[i])
            modified_param = i



    default_value = default_config[modified_param]
    modified_value = modified_config[modified_param]

    if isinstance(default_value, str):
        change = f"{default_value} -> {modified_value}"
    else:
        change_amount = modified_value - default_value
        change = f"{default_value} -> {modified_value} (Change: {change_amount})"

    return modified_param, change


for i in range(27):
    conf = get_configuration(i)
    print(i,get_modified_param_string(conf))