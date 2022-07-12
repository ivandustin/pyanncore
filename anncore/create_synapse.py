from jax.random import normal

def create_synapse(key, shape):
    return normal(key, shape) * 0.1 + 1.5
