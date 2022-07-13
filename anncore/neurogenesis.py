from .create_synapse import create_synapse
from jax.numpy import concatenate
from jax.random import split

def neurogenesis(key, network):
    keys = split(key, len(network))
    return [ concatenate([ synapse[:,:-1], create_synapse(key, shape=(synapse.shape[0],1)), synapse[:,-1:] ], axis=1) for key, synapse in zip(keys, network) ] + [ create_synapse(key, shape=(1,1)) ]
