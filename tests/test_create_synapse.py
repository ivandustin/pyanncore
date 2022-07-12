from anncore import create_synapse
from jax.random import PRNGKey
from jax.numpy import isclose

def test():
    key     = PRNGKey(0)
    synapse = create_synapse(key, shape=(1000,))
    assert isclose(synapse.mean(), 1.5, rtol=0.1)
    assert isclose(synapse.std(),  0.1, rtol=0.1)
