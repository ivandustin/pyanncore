from .create_synapse import create_synapse
from jax.numpy import concatenate
from jax.random import split

def neurogenesis(key, network):
    key_a, key_b = split(key)
    root  = network[0]
    n     = root.shape[0]
    a     = create_synapse(key_a, shape=(n,1))
    b     = create_synapse(key_b, shape=(1,1))
    left  = root[:,:-1]
    right = root[:,-1:]
    root  = concatenate([left, a, right], axis=1)
    return [root] + network[1:] + [b]
