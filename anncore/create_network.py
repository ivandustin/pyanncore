from .synapse import Synapse

def create_network(key, inputs):
    return [ Synapse(key, shape=(inputs, 1)) ]
