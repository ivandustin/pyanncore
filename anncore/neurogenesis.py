from .create_synapse import create_synapse
from jax.numpy import concatenate
from jax.random import split
from .right import right
from .left import left

def neurogenesis(key, network):
    keys = split(key, len(network))
    return [ concatenate([ left(synapse), create_synapse(key, shape=(synapse.shape[0],1)), right(synapse) ], axis=1) for key, synapse in zip(keys, network) ] + [ create_synapse(key, shape=(1,1)) ]
