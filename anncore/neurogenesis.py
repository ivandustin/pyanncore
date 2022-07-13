from jax.numpy import concatenate
from jax.random import split
from .synapse import Synapse
from .insert import insert

def neurogenesis(key, network):
    keys = split(key, len(network))
    return [ insert(Synapse(key, shape=(synapse.shape[0], 1)), synapse) for key, synapse in zip(keys, network) ] + [ Synapse(key, shape=(1, 1)) ]
