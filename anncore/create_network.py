from .create_synapse import create_synapse

def create_network(key, n):
    return [ create_synapse(key, shape=(n,1)) ]
