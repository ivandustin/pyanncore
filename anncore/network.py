from .synapse import Synapse

def Network(key, inputs):
    return [ Synapse(key, shape=(inputs, 1)) ]
